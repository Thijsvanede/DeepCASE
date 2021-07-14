from collections     import Counter
from sklearn.metrics import classification_report
import argformat
import argparse
import json
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
    context_builder.add_argument('-f', '--features'      , type=int  , default=280,          help="maximum number of expected features")
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
    events, context, label, mapping = preprocessor.csv(args.file, nrows=args.max_sequences, verbose=True)
    # Set to device
    events  = events .to(args.device).unsqueeze(1)
    context = context.to(args.device)
    label   = label  .to(args.device)

    events_test   = events [ int(args.train):]
    context_test  = context[ int(args.train):]
    label_test    = label  [ int(args.train):]

    del events
    del context
    del label

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
    deepcase.context_builder = ContextBuilder.load(
        infile = args.load_context,
        device = args.device,
    )

    # Load Interpreter
    deepcase.interpreter = Interpreter.load(
        infile           = args.load_interpreter,
        context_builder  = deepcase.context_builder,
    )

    ########################################################################
    #                    Get mask of unpredicted events                    #
    ########################################################################

    # Predict testing data
    pred_test  = deepcase.interpreter.predict(
        X       = context_test,
        y       = events_test ,
        verbose = args.verbose,
    ).squeeze(1)

    # Get mask of data that form new clusters
    mask = pred_test <= -1

    torch.save(pred_test, 'saves/old/pred_test_initial.save')
    torch.save(mask             , 'saves/old/mask.save')

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

    print("Original")
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

    del pred_test
    #
    # print(mask.shape)
    # print(mask.sum())

    ########################################################################
    #                          Simulate updating                           #
    ########################################################################

    # # Update context_builder
    # deepcase.context_builder.fit(
    #     X          = context_test,
    #     y          = events_test,
    #     batch_size = args.batch_size,
    #     epochs     = args.epochs,
    #     verbose    = args.verbose,
    # )
    #
    # deepcase.context_builder.save('saves/old/context_updated.save')

    mask = torch.load('saves/old/mask.save')

    # Load ContextBuilder
    deepcase.context_builder = ContextBuilder.load(
        infile = 'saves/old/context_updated.save',
        device = args.device,
    )

    # # Create new Interpreter
    # deepcase.interpreter = Interpreter(
    #     context_builder = deepcase.context_builder,
    #     features        = deepcase.interpreter.features,
    #     eps             = deepcase.interpreter.eps,
    #     min_samples     = deepcase.interpreter.min_samples,
    #     threshold       = deepcase.interpreter.threshold,
    # )
    #
    # # Fit Interpreter
    # deepcase.interpreter.fit(
    #     X          = context_test,
    #     y          = events_test ,
    #     score      = label_test  .unsqueeze(1),
    #     func_score = lambda x: torch.max(x, dim=0).values,
    #     verbose    = args.verbose,
    # )
    #
    # deepcase.interpreter.save('saves/interpreter_updated.save')

    deepcase.interpreter = Interpreter.load(
        infile           = 'saves/interpreter_updated.save',
        context_builder  = deepcase.context_builder,
    )

    # Get mask of new predictions
    pred_test = deepcase.interpreter.predict(
        X       = context_test,
        y       = events_test ,
        verbose = args.verbose,
    ).squeeze(1)[mask]

    torch.save(pred_test, 'saves/old/pred_test.save')

    label_test = label_test[mask]

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

    print("Manual part")
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
