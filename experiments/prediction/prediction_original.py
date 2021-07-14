import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '../../deepseq')

from argformat     import StructuredFormatter
from deeplog       import DeepLog
from deepseq       import DeepSeq
from preprocessing import PreprocessLoader
from tiresias      import Tiresias
from utils         import multiprediction_report

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
    group_training.add_argument('-b', '--batch-size', type=int, default=32    , help="batch size")
    group_training.add_argument('-d', '--device'    , type=str, default='auto', help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10    , help="number of epochs to train with")
    group_training.add_argument('-r', '--random'    , action='store_true'     , help="train with random selection")
    group_training.add_argument('-a', '--ratio'     , type=float, default=0.5 , help="training ratio to use")
    group_training.add_argument('-s', '--silent', dest='verbose', action='store_false', help="supress printing progress")

    # Store/load model
    group_model = parser.add_argument_group("Model I/O parameters")
    group_model.add_argument('--load', help="load model from LOAD file")
    group_model.add_argument('--save', help="save model to   SAVE file")

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

    # ########################################################################
    # #                               DeepLog                                #
    # ########################################################################
    # print("DeepLog prediction")
    # print("-"*40)
    #
    # # Create DeepLog instance
    # deeplog = DeepLog(
    #     input_size  = n_features,
    #     hidden_size = 64,
    #     output_size = n_features,
    #     num_layers  = 2
    # ).to(device)
    #
    # # Train DeepLog
    # deeplog.fit(X_train[:, -10:], y_train.reshape(-1),
    #     epochs        = args.epochs,
    #     batch_size    = 128,
    #     learning_rate = 0.01,
    #     criterion     = nn.CrossEntropyLoss(),
    #     variable      = False,
    #     verbose       = args.verbose,
    # )
    #
    # # Predict using DeepLog
    # y_pred, confidence = deeplog.predict(X_test[:, -10:], y_test, k=args.top, verbose=args.verbose)
    # y_pred = y_pred.unsqueeze(1)
    #
    # # Print result
    # multiprediction_report(y_test.cpu(), y_pred.cpu(), decodings.get('threat_name'), y_train.cpu())
    # print()
    #
    # ########################################################################
    # #                               Tiresias                               #
    # ########################################################################
    # print("Tiresias prediction")
    # print("-"*40)
    #
    # # Create Tiresias instance
    # tiresias = Tiresias(
    #     input_size  = n_features,
    #     hidden_size = 128,
    #     output_size = n_features,
    #     k           = 4
    # ).to(device)
    #
    # # Train tiresias
    # tiresias.fit(X_train[:, -20:], y_train.reshape(-1),
    #     epochs        = args.epochs,
    #     batch_size    = 128,
    #     learning_rate = 0.01,
    #     criterion     = nn.NLLLoss(),
    #     variable      = False,
    #     verbose       = args.verbose,
    # )
    #
    # # Predict using tiresias
    # y_pred, confidence = tiresias.predict(X_test[:, -20:], k=args.top, verbose=args.verbose)
    # y_pred = y_pred.unsqueeze(1)
    #
    # # Print result
    # multiprediction_report(y_test.cpu(), y_pred.cpu(), decodings.get('threat_name'), y_train.cpu())
    # print()

    ########################################################################
    #                               DeepSeq                                #
    ########################################################################
    print("DeepSeq prediction")
    print("-"*40)

    # Initialise deepseq
    deepseq = DeepSeq(n_features, context=args.dim_input, device=device)

    # Fit deepseq
    deepseq = deepseq.fit(X_train[:, -5:], y_train,
        batch_size = args.batch_size,
        epochs     = args.epochs,
        verbose    = args.verbose
    )

    # Perform prediction
    y_pred, attention, confidence = deepseq.seq2seq.predict(X_test[:, -5:],
        amount     = 1,
        top        = args.top,
        batch_size = args.batch_size,
        verbose    = args.verbose
    )

    # Print result
    multiprediction_report(y_test.cpu(), y_pred.cpu(), decodings.get('threat_name'), y_train.cpu())
    print()
