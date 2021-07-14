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
sys.path.insert(1, '../../../deepseq/')

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

    maximum = 100000

    # Get relevant handles
    X = data['threat_name']['X'][:maximum]
    y = data['threat_name']['y'][:maximum]
    breach = data['breach']['y'][:maximum]
    impact = data['impact']['y'][:maximum]
    X_time = data['ts_start']['X'][:maximum]
    y_time = data['ts_start']['y'][:maximum]

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

    ################################################################
    #                       Context Builder                        #
    ################################################################

    for size in [1, 2, 5, 10, 20]:

        for time in [60, 3600, 86400, 604800]:
            X_train_ = X_train[:, -size:]
            X_test_  = X_test [:, -size:]

            X_train_[(y_time_train[:, -size:] - X_time_train[:, -size:]) > time] = 0
            X_test_ [(y_time_test [:, -size:] - X_time_test [:, -size:]) > time] = 0

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

            # Fit context_builder
            deepseq.context_builder.fit(X_train_, y_train,
                batch_size = args.batch_size,
                epochs     = 10,
                verbose    = False,
            )

            pred, _ = deepseq.context_builder.predict(X_test_)
            pred = pred.squeeze(1)
            pred_correct = pred.exp()[torch.arange(X_test.shape[0]), y_test.squeeze(1)]
            print("Classified [size={:8}, time={:8}] = {:8}/{:8} = {:.4f}%".format(
                size,
                time,
                (pred_correct >= 0.2).sum(),
                pred_correct.shape[0],
                100*(pred_correct >= 0.2).sum().item()/pred_correct.shape[0],
            ))

            # y_pred = torch.zeros(pred_correct.shape[0], dtype=torch.long, device=y_test.device) - 1
            # y_pred[pred_correct >= 0.2] = y_test.squeeze(1)[pred_correct >= 0.2]
            #
            # cp = classification_report(
            #     y_pred.cpu().numpy(),
            #     y_test.squeeze(1).cpu().numpy(),
            #     digits=4,
            #     output_dict=True,
            #     zero_division = 0,
            # )
            # print("accuracy    : {}".format(cp.get('accuracy')))
            # print("macro avg")
            # for k, v in sorted(cp.get('macro avg').items()):
            #     print("\t '{:10}': {:.4f}".format(k, v))
            # print("weighted avg")
            # for k, v in sorted(cp.get('weighted avg').items()):
            #     print("\t '{:10}': {:.4f}".format(k, v))
            # print()
