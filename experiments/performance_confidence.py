import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, '../deepcase')

from deepcase      import DeepCASE
from preprocessing import PreprocessLoader
from seq2seq       import Seq2seq
from argformat     import StructuredFormatter
from utils         import multiprediction_report
from sklearn       import metrics


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog        = "deepcase.py",
        description = "DeepCASE: providing meta-level contextual analysis of security alerts",
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
    group_output.add_argument('-o', '--dim-output', type=int, default=1, help="length of output sequence")
    group_output.add_argument('-t', '--top'       , type=int, default=1, help="make TOP most likely predictions")

    # Training
    group_training = parser.add_argument_group("Training parameters")
    group_training.add_argument('-b', '--batch-size', type=int, default=32    , help="batch size")
    group_training.add_argument('-d', '--device'    , type=str, default='auto', help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10    , help="number of epochs to train with")
    group_training.add_argument('-r', '--random'    , action='store_true'     ,  help="train with random selection")
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
    data, encodings = loader.load(args.file, args.dim_input, args.dim_output,
                                  args.max, train_ratio=0.5, random=args.random)

    # Get short handles
    X_train = data.get('threat_name').get('train').get('X').to(device)
    y_train = data.get('threat_name').get('train').get('y').to(device)
    X_test  = data.get('threat_name').get('test' ).get('X').to(device)
    y_test  = data.get('threat_name').get('test' ).get('y').to(device)
    sx_test = data.get('severity').get('test').get('X')
    sy_test = data.get('severity').get('test').get('y')

    # Get number of features
    if args.features > 0:
        n_features = args.features
    else:
        n_features = len(torch.cat((X_train.unique(), y_train.unique(), X_test.unique(), y_test.unique())).unique())
    # Get decodings

    decodings = {k: {v2: k2 for k2, v2 in v.items()} for k, v in encodings.items()}

    ########################################################################
    #                       Perform context analysis                       #
    ########################################################################

    # Initialise DeepCASE
    deepcase = DeepCASE(n_features, context=args.dim_input, device=device)

    # Fit DeepCASE
    deepcase = deepcase.fit(X_train, y_train,
        batch_size = args.batch_size,
        epochs     = args.epochs
    )

    # Perform prediction
    y_pred, attention, confidence = deepcase.seq2seq.predict(X_test,
        amount     = args.dim_output,
        top        = 1,
        batch_size = args.batch_size
    )

    y_pred     = y_pred    .reshape(-1)
    confidence = confidence.reshape(-1)

    confidences = list()
    f1s         = list()
    precisions  = list()
    recalls     = list()
    classified  = list()

    # Loop over all confidence scores
    for c in range(0, int(100*confidence.max())):
        c = c/100
        # Get mask
        mask = confidence >= c
        # Compute accuracy
        report = metrics.classification_report(y_test[mask], y_pred[mask], output_dict=True)
        f1        = report.get('weighted avg').get('f1-score')
        precision = report.get('weighted avg').get('precision')
        recall    = report.get('weighted avg').get('recall')
        support   = report.get('weighted avg').get('support')


        # Append
        confidences.append(c)
        f1s        .append(f1)
        precisions .append(precision)
        recalls    .append(recall)
        classified .append(support / y_test.shape[0])

    # Plot
    confidences = np.asarray(confidences)
    f1s         = np.asarray(f1s       )
    precisions  = np.asarray(precisions)
    recalls     = np.asarray(recalls   )
    classified  = np.asarray(classified)

    plt.plot(confidences, f1s       , c='g', label='f1-score')
    plt.plot(confidences, precisions, c='b', label='precision')
    plt.plot(confidences, recalls   , c='r', label='recall')
    plt.plot(confidences, classified, c='k', label='% classified')
    plt.legend()
    plt.show()
    exit()



    print("Predicted {:.2f}%".format(100*confidence_mask.sum()/sy_pred.shape[0]))
    sy_pred = sy_pred[confidence_mask].cpu().numpy().flatten()
    sy_test = sy_test[confidence_mask].cpu().numpy().flatten()

    from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
    print(median_absolute_error(sy_test, sy_pred))
    print(mean_squared_error   (sy_test, sy_pred))
    print(r2_score             (sy_test, sy_pred))

    multiprediction_report(y_test[~confidence_mask].cpu(), y_pred[~confidence_mask].cpu(), decodings.get('threat_name'))
    multiprediction_report(y_test[ confidence_mask].cpu(), y_pred[ confidence_mask].cpu(), decodings.get('threat_name'))
    multiprediction_report(y_test                  .cpu(), y_pred                  .cpu(), decodings.get('threat_name'))
