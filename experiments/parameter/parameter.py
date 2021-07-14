from sklearn.metrics import classification_report
import argparse
import numpy as np
import sys
import torch

sys.path.insert(0, '../../deepseq')

from argformat     import StructuredFormatter
from deepseq       import DeepSeq
from preprocessing import PreprocessLoader

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog        = "deepseq.py",
        description = "DeepSeq: providing meta-level contextual analysis of security alerts",
        formatter_class=StructuredFormatter
    )

    # Input
    group_input = parser.add_argument_group("Input parameters")
    group_input.add_argument('file', help="read preprocessed file")
    group_input.add_argument('-f', '--features' , type=int,   default=275,          help="maximum number of expected features")
    group_input.add_argument('-i', '--dim-input', type=int,   default=5,            help="length of input sequence")
    group_input.add_argument('-m', '--max',       type=float, default=float('inf'), help="maximum number of rows ro read from input")

    # Output
    group_output = parser.add_argument_group("Output parameters")
    group_output.add_argument('-t', '--top'       , type=int, default=1, help="make TOP most likely predictions")

    # Training
    group_training = parser.add_argument_group("Training parameters")
    group_training.add_argument('-b', '--batch-size', type=int, default=128   , help="batch size")
    group_training.add_argument('-d', '--device'    , type=str, default='auto', help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10    , help="number of epochs to train with")
    group_training.add_argument('-r', '--random'    , action='store_true'     , help="train with random selection")
    group_training.add_argument('-a', '--ratio'     , type=float, default=0.5 , help="training ratio to use")
    group_training.add_argument('-s', '--silent', dest='verbose', action='store_false', help="supress printing progress")

    # Parse arguments
    args = parser.parse_args()

    # Set device
    if args.device is None or args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create loader for preprocessed data
    loader = PreprocessLoader()
    # Load data
    data, encodings = loader.load(args.file, 20, 1,
                                  args.max, train_ratio=args.ratio, random=args.random,
                                  extract=['threat_name'],
                                  encode ={'threat_name'})

    # Get short handles
    X_train = data.get('threat_name').get('train').get('X').to(device)
    y_train = data.get('threat_name').get('train').get('y').to(device)
    X_test  = data.get('threat_name').get('test' ).get('X').to(device)
    y_test  = data.get('threat_name').get('test' ).get('y').to(device)

    # Get number of features
    if args.features > 0:
        n_features = args.features
    else:
        n_features = len(torch.cat((X_train.unique(), y_train.unique(), X_test.unique(), y_test.unique())).unique())
    # Get decodings
    decodings = {k: {v2: k2 for k2, v2 in v.items()} for k, v in encodings.items()}

    ########################################################################
    #                               DeepSeq                                #
    ########################################################################
    for complexity in range(12):
        # Set complexity
        complexity = 2**complexity

        for window in range(1, 20):

            # Initialise average performance
            out_pred = list()
            out_test = list()

            for fold in range(10):

                # Initialise deepseq
                deepseq = DeepSeq(n_features, complexity=complexity, context=window, device=device)

                # Fit deepseq
                deepseq = deepseq.fit(X_train[:, -window:], y_train,
                    batch_size = args.batch_size,
                    epochs     = args.epochs,
                    verbose    = args.verbose
                )

                # Perform prediction
                y_pred, attention, confidence = deepseq.seq2seq.predict(X_test[:, -window:],
                    amount     = 1,
                    top        = args.top,
                    batch_size = args.batch_size,
                    verbose    = args.verbose
                )

                # Prepare output
                y_pred_ = y_pred.reshape(-1).cpu().numpy()
                y_test_ = y_test.reshape(-1).cpu().numpy()

                # Append to output
                out_pred.append(y_pred_)
                out_test.append(y_test_)

            # Compute average classification report over 10-fold
            print("Complexity = {}, window = {}".format(complexity, window))
            print("-"*40)
            print(classification_report(
                np.concatenate(out_test),
                np.concatenate(out_pred),
                digits=4,
            ))
            print()
