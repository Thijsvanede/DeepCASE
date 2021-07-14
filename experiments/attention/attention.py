from sklearn.metrics import classification_report
import argparse
import numpy as np
import sys
import torch

sys.path.insert(0, '../../deepseq')

from argformat     import StructuredFormatter
from deepseq       import DeepSeq
from preprocessing import PreprocessLoader
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
    data, encodings = loader.load(args.file, args.dim_input, 1,
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

    full_attention = list()
    full_y_pred    = list()
    full_y_true    = list()

    for fold in range(10):

        # Initialise deepseq
        deepseq = DeepSeq(n_features, complexity=128, context=args.dim_input, device=device)

        # Fit deepseq
        deepseq = deepseq.fit(X_train, y_train,
            batch_size = args.batch_size,
            epochs     = args.epochs,
            verbose    = args.verbose
        )

        # Perform prediction
        y_pred, attention, confidence = deepseq.seq2seq.predict(X_test,
            amount     = 1,
            top        = args.top,
            batch_size = args.batch_size,
            verbose    = args.verbose
        )

        # Reshape attention
        attention = attention.reshape(-1, args.dim_input)
        full_attention.append(attention)
        full_y_pred   .append(y_pred)
        full_y_true   .append(y_test)

    attentions = torch.cat(full_attention)
    full_y_pred = torch.cat(full_y_pred)
    full_y_true = torch.cat(full_y_true)
    # Export to file
    np.save('attention', attentions.cpu().numpy())
    np.save('y_pred', full_y_pred.cpu().numpy())
    np.save('y_true', full_y_true.cpu().numpy())
    exit()
    print(attentions)
    print(attentions.shape)

    import matplotlib.pyplot as plt
    fig = plt.figure()

    # Make a histogram of attentions
    from collections import Counter
    for i, attention in enumerate(torch.unbind(attentions, dim=1)):
        # counts = Counter(X_test.cpu().numpy().flatten())
        # most_common = counts.most_common(1)[0][0]
        # mask = (y_pred != most_common).reshape(-1)

        plt.subplot(args.dim_input, 1, i+1)
        plt.hist(attention.cpu().numpy(), 50, range=[0, 1])

    plt.show()

    print(y_test.shape)
    print(y_pred.shape)
    multiprediction_report(y_test.cpu(), y_pred.cpu(), decodings.get('threat_name'), y_train.cpu())

    exit()
