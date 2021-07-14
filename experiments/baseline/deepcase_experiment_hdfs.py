from collections     import Counter
from sklearn.metrics import classification_report
import argformat
import argparse
import numpy as np
import scipy as sp
import sys
import torch


from deepcase                 import DeepCASE
from deepcase.preprocessing   import Preprocessor
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter     import Interpreter
from deepcase.utils           import confusion_report


if __name__ == "__main__":
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

    # ContextBuilder parameters
    context_builder = parser.add_argument_group("ContextBuilder")
    context_builder.add_argument('-f', '--features'      , type=int  , default=20,          help="maximum number of expected features")
    context_builder.add_argument('-i', '--context'       , type=int  , default=10,           help="length of input sequence")
    context_builder.add_argument('-o', '--dim-output'    , type=int  , default=1,            help="length of output sequence")
    context_builder.add_argument('-m', '--max-sequences' , type=float, default=None        , help="maximum number of sequences ro read from input")
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

    # Create preprocessor
    preprocessor = Preprocessor(
        context = args.context,
        timeout = args.time,
    )

    # Load data from file
    events_train, context_train, label_train, mapping_train = preprocessor.text(args.file, labels=0, verbose=True)
    events_test , context_test , label_test , mapping_test  = preprocessor.text(args.test, labels=0, verbose=True)

    # Ensure mappings are the same
    mapping = np.vectorize(lambda x: mapping_test[x])
    events_test  = mapping(events_test)
    context_test = mapping(context_test)

    mapping_train_inv = {v: k for k, v in mapping_train.items()}
    for unknown in set(mapping_test.values()) - set(mapping_train_inv):
        mapping_train_inv[unknown] = max(mapping_train_inv.values()) + 1
    mapping = np.vectorize(lambda x: mapping_train_inv[x])
    events_test  = mapping(events_test)
    context_test = mapping(context_test)

    events_test  = torch.Tensor(events_test ).to(torch.long)
    context_test = torch.Tensor(context_test).to(torch.long)


    # Set to device
    events_train  = events_train .to(args.device).unsqueeze(1)
    context_train = context_train.to(args.device)
    label_train   = label_train  .to(args.device)

    events_test   = events_test  .to(args.device).unsqueeze(1)
    context_test  = context_test .to(args.device)
    label_test    = label_test   .to(args.device)

    ########################################################################
    #                           Create DeepCASE                            #
    ########################################################################

    # Create DeepCASE object
    deepcase = DeepCASE(
        n_features  = args.features,
        complexity  = args.complexity,
        context     = args.context,
        device      = args.device,
        eps         = args.epsilon,
        min_samples = args.min_samples,
        threshold   = args.threshold,
    )

    ########################################################################
    #                          Fit/Load DeepCASE                           #
    ########################################################################

    # Load ContextBuilder
    if args.load_context:
        deepcase.context_builder = ContextBuilder.load(
            infile = args.load_context,
            device = args.device,
        )
    else:
        # Fit ContextBuilder
        deepcase.context_builder.fit(
            X          = context_train,
            y          = events_train,
            batch_size = args.batch_size,
            epochs     = args.epochs,
            verbose    = args.verbose,
        )

    # Save ContextBuilder if necessary
    if args.save_context:
        deepcase.context_builder.save(args.save_context)

    # Load Interpreter
    if args.load_interpreter:
        deepcase.interpreter = Interpreter.load(
            infile           = args.load_interpreter,
            context_builder  = deepcase.context_builder,
        )
    else:
        # Fit Interpreter
        deepcase.interpreter.fit(
            X          = context_train,
            y          = events_train,
            score      = label_train.unsqueeze(1),
            func_score = lambda x: torch.max(x, dim=0).values,
            verbose    = args.verbose,
        )

    # Save Interpreter if necessary
    if args.save_interpreter:
        deepcase.interpreter.save(args.save_interpreter)

    ########################################################################
    #                      Evaluate DeepCASE - Train                       #
    ########################################################################

    # Predict training data
    pred_train = deepcase.interpreter.predict(
        X       = context_train,
        y       = events_train,
        verbose = args.verbose,
    ).squeeze(1)

    del context_train
    del events_train

    # Transform back to labels
    y_pred_train = np.zeros(pred_train.shape[0])
    y_pred_train[:] = pred_train[:]

    # Set impact - train
    y_pred_train[pred_train >= 0] = 0
    y_pred_train[pred_train >  0] = 1
    y_pred_train[pred_train >  1] = 2
    y_pred_train[pred_train >  2] = 3
    y_pred_train[pred_train >  3] = 4

    pred_train = y_pred_train.astype(int)

    ########################################################################
    #                          Print evaluation                           #
    ########################################################################

    print(classification_report(
        y_true        = label_train[pred_train >= 0].cpu().numpy(),
        y_pred        = pred_train [pred_train >= 0],
        digits        = 4,
        labels        = [0, 1, 2, 3, 4],
        target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        zero_division = 0,
    ))

    print("-"*60)

    print(confusion_report(
        y_true        = label_train.cpu().numpy(),
        y_pred        = pred_train ,
        labels        = [-3, -2, -1, 0, 1, 2, 3, 4],
        target_names  = ['-3', '-2', '-1', 'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
    ))

    print("-"*60)

    print("Unable to predict {}/{} = {:.2f}%".format(
        (pred_train < 0).sum(),
        pred_train.shape[0],
        100 * (pred_train < 0).sum() / pred_train.shape[0],
    ))

    # Calculate statistics - clusters
    clusters = Counter(deepcase.interpreter.clusters)
    # Remove anomaly cluster
    if -1 in clusters: del clusters[-1]
    cluster_counts = np.asarray(list(clusters.values()))

    print("Number of clusters: {}".format(cluster_counts.shape[0]))
    print()

    del pred_train
    del label_train

    ########################################################################
    #                       Evaluate DeepCASE - Test                       #
    ########################################################################

    # Predict testing data
    pred_test  = deepcase.interpreter.predict(
        X       = context_test ,
        y       = events_test ,
        verbose = args.verbose,
    ).squeeze(1)

    del context_test
    del events_test

    # Transform back to labels
    y_pred_test  = np.zeros(pred_test .shape[0])

    y_pred_test [:] = pred_test [:]
    # Set impact - test
    y_pred_test [pred_test  >= 0] = 0
    y_pred_test [pred_test  >  0] = 1
    y_pred_test [pred_test  >  1] = 2
    y_pred_test [pred_test  >  2] = 3
    y_pred_test [pred_test  >  3] = 4

    pred_test  = y_pred_test .astype(int)

    ########################################################################
    #                          Print evaluation                           #
    ########################################################################

    print(classification_report(
        y_true        = label_test[pred_test >= 0].cpu().numpy(),
        y_pred        = pred_test [pred_test >= 0],
        digits        = 4,
        labels        = [0, 1, 2, 3, 4],
        target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        zero_division = 0,
    ))

    print("-"*60)

    print(confusion_report(
        y_true        = label_test.cpu().numpy(),
        y_pred        = pred_test ,
        labels        = [-3, -2, -1, 0, 1, 2, 3, 4],
        target_names  = ['-3', '-2', '-1', 'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
    ))

    from sklearn.metrics import confusion_matrix
    label_indices = np.unique(
        np.unique(label_test.cpu().numpy()).tolist() +
        np.unique(pred_test).tolist()
    ).tolist()
    print(confusion_matrix(
        y_true = label_test.cpu().numpy(),
        y_pred = pred_test,
        labels = label_indices,
    ))
    print(label_indices)

    print("-"*60)

    print("Unable to predict {}/{} = {:.2f}%".format(
        (pred_test < 0).sum(),
        pred_test.shape[0],
        100 * (pred_test < 0).sum() / pred_test.shape[0],
    ))

    # Calculate statistics - clusters
    clusters = Counter(deepcase.interpreter.clusters)
    # Remove anomaly cluster
    if -1 in clusters: del clusters[-1]
    cluster_counts = np.asarray(list(clusters.values()))

    print("Number of clusters: {}".format(cluster_counts.shape[0]))
    print()
