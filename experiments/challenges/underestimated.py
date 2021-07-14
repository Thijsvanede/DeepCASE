from collections     import Counter
from datetime        import datetime, timedelta
from sklearn.metrics import classification_report, homogeneity_score
import argformat
import argparse
import logging
import numpy as np
import pickle
import sys
import torch
sys.path.insert(1, '../../deepseq/')

from deepseq           import DeepSeq
from context_builder   import ContextBuilder
from interpreter       import Interpreter
from preprocessing     import PreprocessLoader
from utils             import box, confusion_report, header
from interpreter.utils import lookup_table

def cluster_summary(label, fingerprints, score='?', size='?', decoding={}):
    """Return a text summary of fingerprints."""
    # Get mean fingerprint
    fingerprint = fingerprints.mean(axis=0)

    # Create summary
    summary = [
        (decoding.get(i, i), value)
        for i, value in enumerate(fingerprint)
        if value > 0.01
    ]

    # Get lengths
    width = max(len(str(k)) for k, _ in summary)
    summary = ["{:{width}}: {:.4f}".format(k, v, width=width) for k, v in summary]

    # Return summary as string
    return "Cluster [{:6}, size={:5}]: {} \n\t{}".format(
        score, size,
        decoding.get(label, label),
        '\n\t'.join(summary)
    )

if __name__ == "__main__":
    # Set logging
    logging.basicConfig(level=logging.DEBUG, filename='logging.log')

    ########################################################################
    #                           Parse arguments                            #
    ########################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog            = "deepseq.py",
        description     = "DeepSeq: providing meta-level contextual analysis of security alerts",
        formatter_class = argformat.StructuredFormatter,
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
    group_input.add_argument('-p', '--pickle', help="treat PICKLE as pickled object")
    group_input.add_argument('--psave', help="save data as pickled object")

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

    ########################################################################
    #                              Set device                              #
    ########################################################################

    # Set device if necessary
    if args.device is None or args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ########################################################################
    #                              Load data                               #
    ########################################################################

    if args.pickle:
        with open(args.pickle, 'rb') as infile:
            tmp = pickle.load(infile)
            data      = tmp.get('data')
            encodings = tmp.get('encodings')

    else:
        # Load data
        data, encodings = PreprocessLoader().load(args.file, args.dim_input, args.time, args.dim_output,
            max_events    = args.max_events,
            max_sequences = args.max_sequences,
            offset        = args.offset,
            train_ratio   = 0,
            random        = args.random,
            extract       = ['ts_start', 'threat_name', 'impact', 'breach'],
            encode        = {'threat_name', 'source'}
        )

        if args.psave:
            with open(args.psave, 'wb') as outfile:
                pickle.dump({
                    'data'     : data,
                    'encodings': encodings,
                }, outfile)

    # Get short handles - data
    X        = data.get('threat_name').get('test').get('X').to(args.device)
    y        = data.get('threat_name').get('test').get('y').to(args.device)
    X_time   = data.get('ts_start').get('test').get('X')
    y_time   = data.get('ts_start').get('test').get('y').squeeze(1)
    X_impact = data.get('impact').get('test').get('X')
    y_impact = data.get('impact').get('test').get('y')
    X_breach = data.get('breach').get('test').get('X')
    y_breach = data.get('breach').get('test').get('y')

    # Set log masks
    mask_log = (y_impact != -1)[:, 0]

    # Ignore negative impact
    X_impact[X_impact < 0] = 0
    y_impact[y_impact < 0] = 0
    X_breach[X_breach < 0] = 0
    y_breach[y_breach < 0] = 0

    ########################################################################
    #                      Split into train and test                       #
    ########################################################################

    # Split into train/test data
    time_start = datetime.fromtimestamp(y_time.min().item())
    time_end   = datetime.fromtimestamp(y_time.max().item())
    time_test  = time_start + timedelta(days=30)

    mask_train = y_time < time_test.timestamp()
    mask_test  = ~mask_train

    # Training data
    X_train        = X[mask_train]
    y_train        = y[mask_train]
    X_impact_train = X_impact[mask_train]
    y_impact_train = y_impact[mask_train]
    X_breach_train = X_breach[mask_train]
    y_breach_train = y_breach[mask_train]
    mask_log_train = mask_log[mask_train]

    ########################################################################
    #                             Get decoding                             #
    ########################################################################

    # Get decoding
    import json
    encoding = {v: int(k) for k, v in encodings['threat_name'].items()}
    with open('{}.encoding.json'.format(args.file), 'r') as infile:
        decoding = {i:k for i, k in enumerate (json.load(infile).get('threat_name'))}

    decoding = {k: decoding.get(v, v) for k, v in encoding.items()}

    ########################################################################
    #                            Create DeepSeq                            #
    ########################################################################

    # Initialise deepseq
    deepseq = DeepSeq(
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
        deepseq.context_builder = ContextBuilder.load(
            infile = args.load_context,
            device = args.device,
        )
    # Fit ContextBuilder
    else:
        deepseq.context_builder.fit(X_train, y_train,
            batch_size = args.batch_size,
            epochs     = args.epochs,
            verbose    = args.verbose,
        )

    # Save ContextBuilder if necessary
    if args.save_context:
        deepseq.context_builder.save(args.save_context)

    # Load Interpreter if necessary
    if args.load_interpreter:
        deepseq.interpreter = Interpreter.load(
            infile          = args.load_interpreter,
            context_builder = deepseq.context_builder,
        )
    # Fit Interpreter
    else:
        # Get scores
        score_y = torch.stack((y_impact_train, y_breach_train), dim=2)

        # Define score function
        def func_score(X):
            # return X.mean(dim=0)
            return X.max(dim=0).values

        deepseq.interpreter.fit(X_train, y_train,
            score      = score_y.squeeze(1),
            func_score = func_score,
            verbose    = args.verbose,
        )

    # Save Interpreter if necessary
    if args.save_interpreter:
        deepseq.interpreter.save(args.save_interpreter)

    ########################################################################
    #                            Analysis Test                             #
    ########################################################################
    time_test = time_start

    # Test data
    mask_test = y_time >= time_test.timestamp()
    X_test        = X       [mask_test]
    y_test        = y       [mask_test]
    X_impact_test = X_impact[mask_test]
    y_impact_test = y_impact[mask_test]
    X_breach_test = X_breach[mask_test]
    y_breach_test = y_breach[mask_test]
    mask_log_test = mask_log[mask_test]

    # Get host info
    host = data.get('host').get('test').get('y')[mask_test]

    # PsExec interaction = 70
    # Potential SMB Brute Force Attack = 14
    mask_event = np.logical_or(
        (y_test == 14).cpu().numpy(),
        (y_test == 70).cpu().numpy(),
    ).flatten()
    mask_breach = (y_breach_test == True).cpu().numpy().flatten()
    mask_full   = np.logical_and(mask_event, mask_breach)

    hosts_u = np.unique(host[mask_full].cpu().numpy())
    for h in hosts_u:
        host_mask = (host == h).cpu().numpy()
        mask_malicious = np.logical_and(host_mask, mask_breach)
        seqs = torch.cat((X_test[mask_malicious], y_test[mask_malicious]), dim=1)
        times = y_time[mask_test][mask_malicious]
        seqs = seqs[times.argsort()].cpu().numpy()

        print("Host: {}".format(h))
        for seq in seqs:
            seq = [decoding.get(s, s) for s in seq]
            print(', '.join("{:18}".format(str(s)[:18]) for s in seq))
        print()

    host = data.get('host').get('test').get('y')

    # Loop over each month of testing data
    while time_test < time_end:
        # Get end of testframe
        time_test_end = time_test + timedelta(days=30)

        # Get mask
        mask_test = (y_time >= time_test    .timestamp()) &\
                    (y_time <  time_test_end.timestamp())

        # Test data
        host_test     = host[mask_test]
        X_test        = X[mask_test]
        y_test        = y[mask_test]
        X_impact_test = X_impact[mask_test]
        y_impact_test = y_impact[mask_test]
        X_breach_test = X_breach[mask_test]
        y_breach_test = y_breach[mask_test]
        mask_log_test = mask_log[mask_test]

        ################################################################
        #                       Perform analysis                       #
        ################################################################
        # Predict train data
        result_test = deepseq.interpreter.predict(
            X_test,
            y_test,
            verbose = args.verbose,
        )

        ################################################################
        #                          Categorize                          #
        ################################################################

        # True train
        # Initialise impact
        y_impact_test  = y_impact_test[:, 0].cpu().numpy()
        y_true_test    = np.zeros(y_impact_test.shape[0], dtype=int)

        # Set impact
        y_true_test[y_impact_test >=  0] = 1
        y_true_test[y_impact_test >= 30] = 2
        y_true_test[y_impact_test >= 70] = 3
        # Set breaches
        y_true_test[y_breach_test[:, 0].cpu().numpy() > 0] = 4
        # Set info
        y_true_test[~mask_log_test.cpu().numpy()] = 0

        # Predicted train
        # Initialise results
        y_pred_test    = np.zeros(result_test.shape[0], dtype=int)
        y_pred_test[:] = result_test[:, 0]

        # Set impact - train
        y_pred_test[result_test[:, 0] >=  0] = 0
        y_pred_test[result_test[:, 0] >=  5] = 1
        y_pred_test[result_test[:, 0] >= 30] = 2
        y_pred_test[result_test[:, 0] >= 70] = 3
        # Set breaches
        y_pred_test[result_test[:, 1] > 0] = 4

        ################################################################
        #                        Automatic mode                        #
        ################################################################

        box("Automatic Mode")
        print()
        print("Timeframe from {} to {}".format(
            str(datetime.fromtimestamp(y_time[mask_test].min())),
            str(datetime.fromtimestamp(y_time[mask_test].max())),
        ))

        header("Confusion matrix")
        print(confusion_report(
            y_true        = y_true_test,
            y_pred        = y_pred_test,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        ))

        for true_value in reversed(range(5)):
            for pred_value in range(true_value):
                print("\nTrue = {}, Pred = {}".format(true_value, pred_value))

                mask = np.logical_and(
                    y_pred_test == pred_value,
                    y_true_test == true_value,
                )

                print("Samples: {}".format(mask.sum()))
                if mask.sum() == 0: continue

                # Compute fingerprints
                clusters, fingerprints, _ = deepseq.interpreter.cluster(
                    X_test[mask], y_test[mask],
                    eps=0.1,
                    min_samples=1,
                    threshold=0.2,
                    iterations=100,
                    batch_size=1024,
                    verbose=False,
                )

                print("Host: {}".format(torch.unique(host_test[mask], return_counts=True)))
                for cluster, indices in lookup_table(clusters):
                    label = torch.unique(y_test[mask][indices])[0].item()
                    print(cluster_summary(label, fingerprints[indices].toarray(), size=indices.shape[0], decoding=decoding))

        time_test = time_test_end
