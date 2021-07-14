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
        raise ValueError("No pickle found!")

    # Get NONE value
    NONE = encodings['threat_name']['-1337']

    # Remove training items
    for key in data.keys():
        data[key] = data[key]['test']

    # Get relevant handles
    X = data['threat_name']['X']
    y = data['threat_name']['y']
    breach = data['breach']['y']
    impact = data['impact']['y']
    X_time = data['ts_start']['X']
    y_time = data['ts_start']['y']

    # Create train and test sets
    mask_train = torch.zeros(X.shape[0], dtype=torch.bool)
    mask_train[:mask_train.shape[0]//2] = True
    mask_test = ~mask_train

    X_train = X[mask_train].to(args.device)
    X_test  = X[mask_test ].to(args.device)
    y_train = y[mask_train].to(args.device)
    y_test  = y[mask_test ].to(args.device)
    breach_train = breach[mask_train]
    breach_test  = breach[mask_test ]
    impact_train = impact[mask_train]
    impact_test  = impact[mask_test ]
    X_time_train = X_time[mask_train]
    X_time_test  = X_time[mask_test ]
    y_time_train = y_time[mask_train]
    y_time_test  = y_time[mask_test ]

    ########################################################################
    #                       Optimize input size/time                       #
    ########################################################################

    # Loop over different input sizes
    for size in [1, 2, 5, 10, 20, 50]:
        X_train_s = X_train[:, -size:]
        X_test_s  = X_test [:, -size:]
        X_time_train_s = X_time_train[:, -size:]
        X_time_test_s  = X_time_test [:, -size:]

        # Loop over different input times
        for time in [60, 3600, 86400, 604800, 2592000]:

            print("Size = {}".format(size))
            print("Time = {}".format(time))

            X_train_t = X_train_s.detach().clone()
            X_test_t  = X_test_s.detach().clone()
            X_train_t[X_time_train_s < y_time_train - time] = NONE
            X_test_t [X_time_test_s  < y_time_test  - time] = NONE

            ################################################################
            #                        Create DeepSeq                        #
            ################################################################

            # Initialise deepseq
            deepseq = DeepSeq(
                n_features  = args.features,
                complexity  = args.complexity,
                context     = size,
                device      = args.device,
                eps         = 1,
                min_samples = args.min_samples,
                threshold   = args.threshold,
            )

            ################################################################
            #                       Context Builder                        #
            ################################################################

            # Fit context_builder
            deepseq.context_builder.fit(X_train_t, y_train,
                batch_size = args.batch_size,
                epochs     = 100,
                verbose    = args.verbose,
            )

            # Save context_builder
            deepseq.context_builder.save("models/context_builder_size_{}_time_{}.save".format(size, time))

            ################################################################
            #                         Interpreter                          #
            ################################################################

            # Get scores
            score_y = torch.stack((impact_train, breach_train), dim=2)

            # Define score function
            def func_score(X):
                # return X.mean(dim=0)
                return X.max(dim=0).values

            deepseq.interpreter.fit(X_train_t, y_train,
                score      = score_y.squeeze(1),
                func_score = func_score,
                verbose    = args.verbose,
            )

            # Save Interpreter
            deepseq.interpreter.save("models/interpreter_size_{}_time_{}.save".format(size, time))

            ################################################################
            #                           Predict                            #
            ################################################################

            # Predict train data
            result = deepseq.interpreter.predict(
                X_test_t,
                y_test,
                verbose=args.verbose,
            )

            ################################################################
            #                          Categorize                          #
            ################################################################

            # True test
            # Initialise impact
            impact_test_ = impact_test[:, 0].cpu().numpy()
            y_true       = np.zeros(impact_test_.shape[0], dtype=int)

            # Set impact
            y_true[impact_test_ >=  0] = 1
            y_true[impact_test_ >= 30] = 2
            y_true[impact_test_ >= 70] = 3
            # Set breaches
            y_true[breach_test[:, 0].cpu().numpy() > 0] = 4
            # Set info
            y_true[impact_test_ == -1] = 0

            # Predicted test
            # Initialise results
            y_pred    = np.zeros(result.shape[0], dtype=int)
            y_pred[:] = result[:, 0]

            # Set impact - train
            y_pred[result[:, 0] >=  0] = 0
            y_pred[result[:, 0] >=  5] = 1
            y_pred[result[:, 0] >= 30] = 2
            y_pred[result[:, 0] >= 70] = 3
            # Set breaches
            y_pred[result[:, 1] > 0] = 4

            ################################################################
            #                         Performance                          #
            ################################################################

            print("Anomalous: {}/{} = {}%".format(
                (y_pred < 0).sum(),
                y_pred.shape[0],
                100*(y_pred < 0).sum() / y_pred.shape[0],
            ))

            print(classification_report(
                y_true        = y_true[y_pred >= 0],
                y_pred        = y_pred[y_pred >= 0],
                digits        = 4,
                labels        = [0, 1, 2, 3, 4],
                target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
                zero_division = 0,
            ))

            # Clean up
            del X_train_t
            del X_test_t
        # Clean up
        del X_train_s
        del X_test_s
        del X_time_train_s
        del X_time_test_s
