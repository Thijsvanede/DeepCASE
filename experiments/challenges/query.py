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
from preprocessing     import PreprocessLoader, SimpleLoader
from utils             import box, confusion_report, header
from interpreter.utils import lookup_table

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

        X_impact_train = torch.zeros(X_train.shape, dtype=torch.long)
        y_impact_train = torch.zeros(y_train.shape, dtype=torch.long)
        X_impact_test  = torch.zeros(X_test .shape, dtype=torch.long)
        y_impact_test  = torch.zeros(y_test .shape, dtype=torch.long)

        if args.malicious:
            X_test_m, y_test_m = loader.load(args.malicious, args.dim_input, args.dim_output, args.max_events)
            X_test_m = X_test_m.to(args.device)
            y_test_m = y_test_m.to(args.device)
            X_impact_test_m = torch.full(X_test_m.shape, 100, dtype=torch.long)
            y_impact_test_m = torch.full(y_test_m.shape, 100, dtype=torch.long)

            # Concatenate
            X_test = torch.cat((X_test, X_test_m))
            y_test = torch.cat((y_test, y_test_m))
            X_impact_test = torch.cat((X_impact_test, X_impact_test_m))
            y_impact_test = torch.cat((y_impact_test, y_impact_test_m))

        X = X_test
        y = y_test

    else:

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
        X = data.get('threat_name').get('test').get('X').to(args.device)
        y = data.get('threat_name').get('test').get('y').to(args.device)

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

    ########################################################################
    #                                Query                                 #
    ########################################################################

    threshold = 0.2

    _, _, inverse, original, optimized = deepseq.context_builder.query(X, y,
        iterations          = 100,
        batch_size          = 1024,
        return_optimization = threshold,
        verbose             = True,
    )

    print(np.logical_and(original.cpu().numpy(), ~optimized.cpu().numpy()).sum())

    print("Unique")
    original_  = original .sum().item()
    optimized_ = optimized.sum().item()
    shape = torch.unique(inverse).shape[0]

    print("Original  samples >= {}: {}/{} = {:6.2f}%".format(
        threshold, original_ , shape, 100*original_ /shape
    ))

    print("Optimized samples >= {}: {}/{} = {:6.2f}%".format(
        threshold, optimized_, shape, 100*optimized_/shape
    ))
    print()

    print("Overall")
    original_  = original [inverse].sum().item()
    optimized_ = optimized[inverse].sum().item()

    print("Original  samples >= {}: {}/{} = {:6.2f}%".format(
        threshold, original_ , X.shape[0], 100*original_ /X.shape[0]
    ))

    print("Optimized samples >= {}: {}/{} = {:6.2f}%".format(
        threshold, optimized_, X.shape[0], 100*optimized_/X.shape[0]
    ))
