#Imports
import argparse
import logging
import numpy as np
import torch
from argformat       import StructuredFormatter
from collections     import Counter
from sklearn.metrics import classification_report, homogeneity_score

# Custom imports
from deepcase                   import DeepCASE
from deepcase.context_builder   import ContextBuilder
from deepcase.interpreter       import Interpreter
from deepcase.interpreter.utils import lookup_table
from deepcase.preprocessing_old import PreprocessLoader, SimpleLoader, NONE
from deepcase.utils             import multiprediction_report
from deepcase.utils             import box, header, confusion_report

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, filename='logging.log')
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog        = "deepcase.py",
        description = "DeepCASE: providing contextual analysis of security alerts",
        formatter_class=StructuredFormatter
    )

    # Input
    group_input = parser.add_argument_group("Input")
    group_input.add_argument('file'      ,                            help="read preprocessed input     file")
    group_input.add_argument('test'      , nargs='?' ,                help="read preprocessed test      file")
    group_input.add_argument('malicious' , nargs='?' ,                help="read preprocessed malicious file")
    group_input.add_argument('--offset'  , type=float, default=0    , help="offset for items to load")
    group_input.add_argument('--time'    , type=float, default=86400, help="max time length of input sequence")
    group_input.add_argument('--all'     , dest='all'     , action='store_true', help="perform experiment on all data")
    group_input.add_argument('--breach'  , dest='breach'  , action='store_true', help="read breaches")
    group_input.add_argument('--incident', dest='incident', action='store_true', help="read incidents")
    group_input.add_argument('--ignore'  , dest='ignore'  , action='store_true', help="ignore incident and breach info")

    # ContextBuilder parameters
    context_builder = parser.add_argument_group("ContextBuilder")
    context_builder.add_argument('-f', '--features'      , type=int  , default=280,          help="maximum number of expected features")
    context_builder.add_argument('-i', '--dim-input'     , type=int  , default=10,           help="length of input sequence")
    context_builder.add_argument('-o', '--dim-output'    , type=int  , default=1,            help="length of output sequence")
    context_builder.add_argument('-m', '--max-sequences' , type=float, default=float('inf'), help="maximum number of sequences ro read from input")
    context_builder.add_argument('-n', '--max-events'    , type=float, default=float('inf'), help="maximum number of events to read from input")
    context_builder.add_argument('-c', '--complexity'    , type=int  , default=128,          help="complexity of the model")

    # Training
    group_training = parser.add_argument_group("ContextBuilder training")
    group_training.add_argument('-b', '--batch-size', type=int, default=128   , help="batch size")
    group_training.add_argument('-d', '--device'    , type=str, default='auto', help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10    , help="number of epochs to train with")
    group_training.add_argument('-r', '--random'    , action='store_true'     , help="train with random selection")
    group_training.add_argument('-s', '--silent', dest='verbose', action='store_false', help="supress printing progress")
    group_training.add_argument('--train', type=float, default=0.5, help="training samples to use (or ratio of if 0 <= TRAIN <= 1)")

    # Interpreter parameters
    interpreter = parser.add_argument_group("Interpreter")
    interpreter.add_argument('--epsilon'    , type=float, default=0.1, help="interpreter epsilon     for clustering")
    interpreter.add_argument('--min-samples', type=int,   default=5,   help="interpreter min_samples for clustering")
    interpreter.add_argument('--threshold'  , type=float, default=0.2, help="interpreter confidence threshold for fingerprinting")

    # Store/load model
    group_model = parser.add_argument_group("Model I/O parameters")
    group_model.add_argument('--load-context'    , help="load context builder from LOAD file")
    group_model.add_argument('--load-interpreter', help="load interpreter     from LOAD file")
    group_model.add_argument('--save-context'    , help="save context builder to   SAVE file")
    group_model.add_argument('--save-interpreter', help="save interpreter     to   SAVE file")

    # Parse arguments
    args = parser.parse_args()

    # Set device if necessary
    if args.device is None or args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ########################################################################
    #                              Load data                               #
    ########################################################################

    # Load data - HDFS
    if args.test:
        loader = SimpleLoader()
        X_train, y_train = loader.load(args.file, args.dim_input, args.dim_output, args.max_events)
        X_test , y_test  = loader.load(args.test, args.dim_input, args.dim_output, args.max_events)
        # Cast to device
        X_train = X_train.to(args.device)
        y_train = y_train.to(args.device)
        X_test  = X_test .to(args.device)
        y_test  = y_test .to(args.device)
        encodings = dict()

        X_train_incident = torch.zeros(X_train.shape, dtype=torch.long)
        y_train_incident = torch.zeros(y_train.shape, dtype=torch.long)
        X_test_incident  = torch.zeros(X_test .shape, dtype=torch.long)
        y_test_incident  = torch.zeros(y_test .shape, dtype=torch.long)

        if args.malicious:
            X_test_m, y_test_m = loader.load(args.malicious, args.dim_input, args.dim_output, args.max_events)
            X_test_m = X_test_m.to(args.device)
            y_test_m = y_test_m.to(args.device)
            X_test_incident_m = torch.full(X_test_m.shape, 100, dtype=torch.long)
            y_test_incident_m = torch.full(y_test_m.shape, 100, dtype=torch.long)

            # Concatenate
            X_test = torch.cat((X_test, X_test_m))
            y_test = torch.cat((y_test, y_test_m))
            X_test_incident = torch.cat((X_test_incident, X_test_incident_m))
            y_test_incident = torch.cat((y_test_incident, y_test_incident_m))

    # Load data - Company
    else:
        if args.ignore:
            breach   = None
            incident = None
            extract  = ['threat_name']
        elif args.breach:
            breach   = 'impact_breach'
            incident = None
            extract = ['threat_name', breach]
        elif args.incident:
            breach   = None
            incident = 'incident_impact'
            extract  = ['threat_name', incident]
        else:
            breach   = 'breach'
            incident = 'impact'
            extract  = ['threat_name', incident, breach]

        # Create loader for preprocessed data
        loader = PreprocessLoader()

        # Load data
        data, encodings = loader.load(args.file, args.dim_input, args.time, args.dim_output,
            max_events    = args.max_events,
            max_sequences = args.max_sequences,
            offset        = args.offset,
            train_ratio   = args.train,
            random        = args.random,
            extract       = extract,
            # extract       = ['threat_name', '_id', 'severity', 'confidence', 'impact_incident', 'impact_breach', 'source'],
            encode        = {'threat_name', 'source'}
        )

        # Get short handles - data
        X_train = data.get('threat_name').get('train').get('X').to(args.device)
        y_train = data.get('threat_name').get('train').get('y').to(args.device)
        X_test  = data.get('threat_name').get('test' ).get('X').to(args.device)
        y_test  = data.get('threat_name').get('test' ).get('y').to(args.device)

        # Get short handles - breach
        if breach is not None:
            X_train_breach = data.get(breach).get('train').get('X')
            y_train_breach = data.get(breach).get('train').get('y')
            X_test_breach  = data.get(breach).get('test' ).get('X')
            y_test_breach  = data.get(breach).get('test' ).get('y')

            # Set log masks
            mask_log_train = (y_train_breach != -1)[:, 0]
            mask_log_test  = (y_test_breach  != -1)[:, 0]

            # Set known masks
            mask_known_train = ~((torch.cat((X_train_breach, y_train_breach), dim=1) == -2).any(dim=1))
            mask_known_test  = ~((torch.cat((X_test_breach , y_test_breach ), dim=1) == -2).any(dim=1))

            # Set negative breaches to 0
            X_train_breach[X_train_breach < 0] = 0
            y_train_breach[y_train_breach < 0] = 0
            X_test_breach [X_test_breach  < 0] = 0
            y_test_breach [y_test_breach  < 0] = 0
        else:
            X_train_breach = torch.zeros(X_train.shape, dtype=torch.long)
            y_train_breach = torch.zeros(y_train.shape, dtype=torch.long)
            X_test_breach  = torch.zeros(X_test.shape, dtype=torch.long)
            y_test_breach  = torch.zeros(y_test.shape, dtype=torch.long)

            # Set log masks
            mask_log_train = torch.zeros(y_train.shape[0], dtype=bool, device=args.device)
            mask_log_test  = torch.zeros(y_test .shape[0], dtype=bool, device=args.device)

            # Set known masks
            mask_known_train = torch.ones(X_train.shape[0], dtype=torch.bool)
            mask_known_test  = torch.ones(X_test .shape[0], dtype=torch.bool)

        # Get short handles - incident
        if incident is not None:
            X_train_incident = data.get(incident).get('train').get('X')
            y_train_incident = data.get(incident).get('train').get('y')
            X_test_incident  = data.get(incident).get('test' ).get('X')
            y_test_incident  = data.get(incident).get('test' ).get('y')

            # Set log masks
            mask_log_train = (y_train_incident != -1)[:, 0]
            mask_log_test  = (y_test_incident  != -1)[:, 0]

            # Set negative incidents to 0
            X_train_incident[X_train_incident < 0] = 0
            y_train_incident[y_train_incident < 0] = 0
            X_test_incident [X_test_incident  < 0] = 0
            y_test_incident [y_test_incident  < 0] = 0
        else:
            X_train_incident = torch.zeros(X_train.shape, dtype=torch.long)
            y_train_incident = torch.zeros(y_train.shape, dtype=torch.long)
            X_test_incident  = torch.zeros(X_test.shape, dtype=torch.long)
            y_test_incident  = torch.zeros(y_test.shape, dtype=torch.long)

        # Remove data because it is no longer required
        del data

        # # Mask -100 unknown
        # mask_train = (X_impact_train == -100).any(dim=1) | (y_impact_train == -100).any(dim=1)
        # X_train = X_train[~mask_train]
        # y_train = y_train[~mask_train]
        # X_impact_train = X_impact_train[~mask_train]
        # y_impact_train = y_impact_train[~mask_train]
        #
        # mask_test =  (X_impact_test == -100).any(dim=1) | (y_impact_test == -100).any(dim=1)
        # X_test = X_test[~mask_test]
        # y_test = y_test[~mask_test]
        # X_impact_test = X_impact_test[~mask_test]
        # y_impact_test = y_impact_test[~mask_test]
        #
        # # Set -1 to 0
        # X_impact_train[X_impact_train <= -1] = 0
        # y_impact_train[y_impact_train <= -1] = 0
        # X_impact_test [X_impact_test  <= -1] = 0
        # y_impact_test [y_impact_test  <= -1] = 0

    if not args.all:
        # Get known train samples
        X_train          = X_train         [mask_known_train]
        y_train          = y_train         [mask_known_train]
        X_train_breach   = X_train_breach  [mask_known_train]
        y_train_breach   = y_train_breach  [mask_known_train]
        mask_log_train   = mask_log_train  [mask_known_train]
        X_train_incident = X_train_incident[mask_known_train]
        y_train_incident = y_train_incident[mask_known_train]

        # Get known test samples
        X_test          = X_test         [mask_known_test]
        y_test          = y_test         [mask_known_test]
        X_test_breach   = X_test_breach  [mask_known_test]
        y_test_breach   = y_test_breach  [mask_known_test]
        mask_log_test   = mask_log_test  [mask_known_test]
        X_test_incident = X_test_incident[mask_known_test]
        y_test_incident = y_test_incident[mask_known_test]

    # Automatically extract number of features if necessary
    if args.features <= 0:
         args.features = len(torch.cat((
            X_train.unique(),
            y_train.unique(),
            X_test.unique(),
            y_test.unique()
        )).unique())

    # # Get decodings
    # decodings = {k: {v2: k2 for k2, v2 in v.items()} for k, v in encodings.items()}
    #
    # if 'threat_name' not in decodings:
    #     decodings['threat_name'] = dict()
    #
    # decoding = {k: int(v) for k, v in decodings['threat_name'].items()}
    # import json
    # with open('../data/fast/breaches.csv.encoding.json', 'r') as infile:
    #     decoding2 = json.load(infile)
    # decoding2 = {i: k for i, k in enumerate(decoding2.get('threat_name'))}
    # decoding = {k: decoding2.get(v, v) for k, v in decoding.items()}

    ########################################################################
    #                           Create DeepCASE                            #
    ########################################################################

    # Initialise DeepCASE
    deepcase = DeepCASE(
        n_features  = args.features,
        complexity  = args.complexity,
        context     = args.dim_input,
        device      = args.device,
        eps         = args.epsilon,
        min_samples = args.min_samples,
        threshold   = args.threshold,
    )

    ########################################################################
    #                              Load/save                               #
    ########################################################################

    # Load ContextBuilder if necessary
    if args.load_context:
        deepcase.context_builder = ContextBuilder.load(
            infile = args.load_context,
            device = args.device,
        )
    # Fit ContextBuilder
    else:
        deepcase.context_builder.fit(X_train, y_train,
            batch_size = args.batch_size,
            epochs     = args.epochs,
            verbose    = args.verbose,
        )

    # Save ContextBuilder if necessary
    if args.save_context:
        deepcase.context_builder.save(args.save_context)

    # Load Interpreter if necessary
    if args.load_interpreter:
        deepcase.interpreter = Interpreter.load(
            infile           = args.load_interpreter,
            context_builder  = deepcase.context_builder,
        )
    # Fit Interpreter
    else:
        score = torch.stack((y_train_incident, y_train_breach), dim=2)
        deepcase.interpreter.fit(X_train, y_train,
            score   = score.squeeze(1),
            verbose = args.verbose,
        )

    # Save Interpreter if necessary
    if args.save_interpreter:
        deepcase.interpreter.save(args.save_interpreter)

    ########################################################################
    #                       Delete unused variables                        #
    ########################################################################
    del X_train_incident
    del X_train_breach

    ########################################################################
    #                               Analyses                               #
    ########################################################################


    # # Perform context analysis
    # y_pred, confidence = deepcase.context_builder.forecast(X_test,
    #     batch_size = args.batch_size,
    # )
    #
    # print("\nPrediction:")
    # multiprediction_report(y_test.cpu(), y_pred.cpu(), decoding, y_train.cpu(), min_train=1)
    #
    # if torch.cuda.is_available():
    #     del y_pred, confidence
    #     torch.cuda.empty_cache()

    # # Perform explainability experiment
    # y_pred, confidence, attention = deepcase.interpreter.explain(X_test, y_test)
    # y_pred = y_pred.unsqueeze(1).unsqueeze(2)
    # print("\n\nExplainable:")
    # multiprediction_report(y_test.cpu(), y_pred.cpu(), decodings.get('threat_name'), y_train.cpu(), min_train=1)
    #
    #
    # if torch.cuda.is_available():
    #     del y_pred, attention
    #     torch.cuda.empty_cache()

    # Compute prediction of both training and testing
    result_train = deepcase.interpreter.predict(X_train, y_train, verbose=args.verbose)
    del X_train
    del y_train
    result_test  = deepcase.interpreter.predict(X_test , y_test , verbose=args.verbose)
    del X_test
    del y_test

    ########################################################################
    #                          Categorize actual                           #
    ########################################################################
    # Cast to numpy
    y_train_incident = y_train_incident[:, 0].cpu().numpy()
    y_test_incident  = y_test_incident [:, 0].cpu().numpy()

    # Initialise results
    y_true_train = np.zeros(y_train_incident.shape[0])
    y_true_test  = np.zeros(y_test_incident .shape[0])
    y_true_train[:] = y_train_incident
    y_true_test [:] = y_test_incident

    # Set impact - train
    y_true_train[y_train_incident >=  0] = 1
    y_true_train[y_train_incident >= 30] = 2
    y_true_train[y_train_incident >= 70] = 3
    # Set impact - test
    y_true_test [y_test_incident  >=  0] = 1
    y_true_test [y_test_incident  >= 30] = 2
    y_true_test [y_test_incident  >= 70] = 3

    # Set breaches
    y_true_train[y_train_breach[:, 0].cpu().numpy() > 0] = 4
    y_true_test [y_test_breach [:, 0].cpu().numpy() > 0] = 4

    # Set INFO
    y_true_train[~mask_log_train.cpu().numpy()] = 0
    y_true_test [~mask_log_test .cpu().numpy()] = 0

    ########################################################################
    #                          Categorize output                           #
    ########################################################################

    # Initialise results
    y_pred_train = np.zeros(result_train.shape[0])
    y_pred_test  = np.zeros(result_test .shape[0])
    y_pred_train[:] = result_train[:, 0]
    y_pred_test [:] = result_test [:, 0]

    # Set impact - train
    y_pred_train[result_train[:, 0] >=  0] = 0
    y_pred_train[result_train[:, 0] >=  5] = 1
    y_pred_train[result_train[:, 0] >= 30] = 2
    y_pred_train[result_train[:, 0] >= 70] = 3
    # Set impact - test
    y_pred_test [result_test [:, 0] >=  0] = 0
    y_pred_test [result_test [:, 0] >=  5] = 1
    y_pred_test [result_test [:, 0] >= 30] = 2
    y_pred_test [result_test [:, 0] >= 70] = 3

    # Set breaches
    y_pred_train[result_train[:, 1] > 0] = 4
    y_pred_test [result_test [:, 1] > 0] = 4

    ########################################################################
    #                             Manual Mode                              #
    ########################################################################
    print("\n\n")
    box("Manual Mode")

    # Calculate statistics - datapoints
    datapoints  = len(deepcase.interpreter.clusters)
    clustered   = (deepcase.interpreter.clusters != -1).sum()
    anomalies   = (deepcase.interpreter.clusters == -1).sum()
    anomalies_c = (result_train == -1).sum()
    width       = len(str(datapoints))

    # Calculate statistics - clusters
    clusters = Counter(deepcase.interpreter.clusters)
    # Remove anomaly cluster
    if -1 in clusters: del clusters[-1]
    cluster_counts = np.asarray(list(clusters.values()))

    # Print results
    header("Statistics - Datapoints")
    print("Datapoints            : {:{width}}".format(datapoints, width=width))
    print("Clustered             : {:{width}}/{:{width}} = {:6.2f}%".format(clustered  , datapoints, 100*clustered  /datapoints, width=width))
    print("Anomalies             : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies  , datapoints, 100*anomalies  /datapoints, width=width))
    print("Anomalies < confidence: {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_c, datapoints, 100*anomalies_c/datapoints, width=width))
    print()

    header("Statistics - Clusters")
    print("Labels              : {}".format(len(deepcase.interpreter.tree)))
    print("Clusters            : {}".format(len(clusters)))
    print("Cluster size avarage: {:.4f}".format(cluster_counts.mean()))
    print("Cluster size std    : {:.4f}".format(cluster_counts.std ()))
    print("Cluster size min    : {}"    .format(cluster_counts.min ()))
    print("Cluster size max    : {}"    .format(cluster_counts.max ()))
    print()

    ####################################################################
    #                           Performance                            #
    ####################################################################

    header("Performance")
    mask = y_pred_train >= 0
    print(classification_report(
        y_true        = y_true_train[mask],
        y_pred        = y_pred_train[mask],
        digits        = 4,
        labels        = [0, 1, 2, 3, 4],
        target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        zero_division = 0,
    ))

    header("Confusion matrix")
    print(confusion_report(
        y_true        = y_true_train[mask],
        y_pred        = y_pred_train[mask],
        labels        = [0, 1, 2, 3, 4],
        target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
    ))

    ####################################################################
    #                         Cluster metrics                          #
    ####################################################################

    header("Cluster metrics")
    scores   = deepcase.interpreter.scores
    clusters = deepcase.interpreter.clusters[deepcase.interpreter.clusters != -1]
    scores_c = np.zeros(scores.shape[0])
    scores_c[scores[:, 0] ==  0] = 0
    scores_c[scores[:, 0] >   0] = 1
    scores_c[scores[:, 0] >= 30] = 2
    scores_c[scores[:, 0] >= 70] = 3
    scores_c[scores[:, 1] >   0] = 4

    # Keep stats
    first = True
    full_info   = 0
    full_low    = 0
    full_medium = 0
    full_high   = 0
    full_attack = 0
    mixed       = 0

    for cluster, indices in sorted(lookup_table(clusters, verbose=False)):
        # Compute categories
        info    = (scores_c[indices] == 0).sum()
        low     = (scores_c[indices] == 1).sum()
        medium  = (scores_c[indices] == 2).sum()
        high    = (scores_c[indices] == 3).sum()
        attacks = (scores_c[indices] == 4).sum()

        if info == indices.shape[0]:
            full_info += 1
        elif low == indices.shape[0]:
            full_low += 1
        elif medium == indices.shape[0]:
            full_medium += 1
        elif high == indices.shape[0]:
            full_high += 1
        elif attacks == indices.shape[0]:
            full_attack += 1
        # Mixed
        else:
            mixed += 1
            if first:
                print("{:30} {:>8} {:>8} {:>8} {:>8} {:>8}".format("MIXED CLUSTERS", "INFO", "LOW", "MEDIUM", "HIGH", "ATTACK"))
                first = False
            print("Cluster {:5} [size={:8}]: {:8} {:8} {:8} {:8} {:8}".format(
                cluster, indices.shape[0], info, low, medium, high, attacks))

    if not first: print()

    n_clusters = np.unique(clusters).shape[0]
    width = len(str(n_clusters))
    print("Clusters           : {}".format(n_clusters))
    print("Clusters INFO      : {:>{width}}".format(full_info  , width=width))
    print("Clusters LOW       : {:>{width}}".format(full_low   , width=width))
    print("Clusters MEDIUM    : {:>{width}}".format(full_medium, width=width))
    print("Clusters HIGH      : {:>{width}}".format(full_high  , width=width))
    print("Clusters ATTACK    : {:>{width}}".format(full_attack, width=width))
    print("Clusters MIXED     : {:>{width}}".format(mixed      , width=width))
    print()

    print("Homogeneity attacks: {:.4f}".format(homogeneity_score(scores[:, 1] > 0, clusters)))
    print("Homogeneity overall: {:.4f}".format(homogeneity_score(scores_c, clusters)))
    print("\n\n")

    ########################################################################
    #                            Automatic Mode                            #
    ########################################################################
    box("Automatic Mode")
    print()

    header("Statistics - Workload Reduction")
    datapoints = y_pred_test.shape[0]
    automated  = (y_pred_test >= 0).sum()
    anomalous  = (y_pred_test <  0).sum()
    anomalies_conf  = (y_pred_test == -1).sum()
    anomalies_train = (y_pred_test == -2).sum()
    anomalies_eps   = (y_pred_test == -3).sum()
    width = len(str(datapoints))
    print("Datapoints            : {:{width}}".format(datapoints, width=width))
    print("Automated             : {:{width}}/{:{width}} = {:6.2f}%".format(automated, datapoints, 100*automated/datapoints, width=width))
    print("Anomalous             : {:{width}}/{:{width}} = {:6.2f}%".format(anomalous, datapoints, 100*anomalous/datapoints, width=width))
    print("Anomalous confidence  : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_conf, datapoints, 100*anomalies_conf/datapoints, width=width))
    print("Anomalous not in train: {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_train, datapoints, 100*anomalies_train/datapoints, width=width))
    print("Anomalous > epsilon   : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_eps, datapoints, 100*anomalies_eps/datapoints, width=width))
    print()

    header("Statistics - Anomalies")
    y_true_anomalous = y_true_test[y_pred_test < 0]
    y_pred_anomalous = y_pred_test[y_pred_test < 0]

    print(confusion_report(
        y_true = y_true_anomalous,
        y_pred = y_pred_anomalous,
        labels = [-3, -2, -1, 0, 1, 2, 3, 4],
        target_names  = ['CONF', 'TRAIN', 'EPS',
                         'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        skip_x = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        skip_y = ['CONF', 'TRAIN', 'EPS'],
    ))

    header("Performance")
    mask = y_pred_test >= 0
    print(classification_report(
        y_true        = y_true_test[mask],
        y_pred        = y_pred_test[mask],
        digits        = 4,
        labels        = [0, 1, 2, 3, 4],
        target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        zero_division = 0,
    ))

    header("Confusion matrix")
    print(confusion_report(
        y_true        = y_true_test[mask],
        y_pred        = y_pred_test[mask],
        labels        = [0, 1, 2, 3, 4],
        target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
    ))
