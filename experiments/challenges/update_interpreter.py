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
    group_input.add_argument('--timedelta', type=int, default=30, help="number of days per update")

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
    #                            Load original                             #
    ########################################################################

    deepseq.context_builder = ContextBuilder.load(
        infile = '../../models/context/company_5m.save',
        device = args.device,
    )

    deepseq.interpreter = Interpreter.load(
        infile          = 'interpreter_5m.save',
        context_builder = deepseq.context_builder,
    )

    ########################################################################
    #                            Analysis Test                             #
    ########################################################################

    month = 1
    interpreters = dict()
    # Loop over each month of testing data
    while time_test < time_end:
        # Get end of testframe
        time_test_end = time_test + timedelta(days=args.timedelta)

        # Get mask
        mask_test  = (y_time >= time_test    .timestamp()) &\
                     (y_time <  time_test_end.timestamp())

        # Test data
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

        result_test_orig = deepseq.interpreter.predict(
            X_test,
            y_test,
            verbose=args.verbose
        )
        result_test_updated = np.zeros(result_test_orig.shape)
        result_test_updated[:] = result_test_orig

        for m, interpreter in sorted(interpreters.items()):
            #  Get mask
            mask_to_predict = np.any(result_test_updated <= -2, axis=1)
            if not np.any(mask_to_predict):
                break

            print("Predicting interpreter {}: {} samples".format(m, mask_to_predict.sum()))

            # Predict test data
            result_test_updated[mask_to_predict] = interpreter.predict(
                X_test[mask_to_predict],
                y_test[mask_to_predict],
                verbose = args.verbose,
            )

        # Retrain interpreter with anomalous datapoints
        mask_retrain = np.any(result_test_updated <= -2, axis=1)
        if np.any(mask_retrain):

            X_retrain = X_test[mask_retrain]
            y_retrain = y_test[mask_retrain]

            # Create new interpreter
            interpreters[month] = Interpreter(deepseq.context_builder, args.features,
                eps         = args.epsilon,
                min_samples = 1,
                threshold   = args.threshold
            )

            # Fit new interpreter
            # Get scores
            score_X = torch.stack((X_impact_test[mask_retrain], X_breach_test[mask_retrain]), dim=2)
            score_y = torch.stack((y_impact_test[mask_retrain], y_breach_test[mask_retrain]), dim=2)

            # Define score function
            def func_score(X):
                # return X.mean(dim=0)
                return X.max(dim=0).values

            interpreters[month].fit(X_retrain, y_retrain,
                score_X = score_X,
                score_y = score_y.squeeze(1),
                func_score = func_score,
                verbose = args.verbose,
            )

            result_test_updated_before = np.zeros(result_test_updated.shape)
            result_test_updated_before[:] = result_test_updated
            result_test_updated_before[mask_retrain] = interpreters[month].predict(
                X_retrain,
                y_retrain,
                verbose=args.verbose
            )

        ################################################################
        #                          Categorize                          #
        ################################################################

        # True train
        # Initialise impact
        y_impact_test = y_impact_test[:, 0].cpu().numpy()
        y_true_test   = np.zeros(y_impact_test.shape[0], dtype=int)

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
        y_pred_test_orig    = np.zeros(result_test_orig.shape[0], dtype=int)
        y_pred_test_orig[:] = result_test_orig[:, 0]

        # Set impact - train
        y_pred_test_orig[result_test_orig[:, 0] >=  0] = 0
        y_pred_test_orig[result_test_orig[:, 0] >=  5] = 1
        y_pred_test_orig[result_test_orig[:, 0] >= 30] = 2
        y_pred_test_orig[result_test_orig[:, 0] >= 70] = 3
        # Set breaches
        y_pred_test_orig[result_test_orig[:, 1] > 0] = 4

        y_pred_test_updated    = np.zeros(result_test_updated.shape[0], dtype=int)
        y_pred_test_updated[:] = result_test_updated[:, 0]

        # Set impact - train
        y_pred_test_updated[result_test_updated[:, 0] >=  0] = 0
        y_pred_test_updated[result_test_updated[:, 0] >=  5] = 1
        y_pred_test_updated[result_test_updated[:, 0] >= 30] = 2
        y_pred_test_updated[result_test_updated[:, 0] >= 70] = 3
        # Set breaches
        y_pred_test_updated[result_test_updated[:, 1] > 0] = 4



        y_pred_test_updated_before = np.zeros(result_test_updated_before.shape[0], dtype=int)
        y_pred_test_updated_before[:] = result_test_updated_before[:, 0]

        # Set impact - train
        y_pred_test_updated_before[result_test_updated_before[:, 0] >=  0] = 0
        y_pred_test_updated_before[result_test_updated_before[:, 0] >=  5] = 1
        y_pred_test_updated_before[result_test_updated_before[:, 0] >= 30] = 2
        y_pred_test_updated_before[result_test_updated_before[:, 0] >= 70] = 3
        # Set breaches
        y_pred_test_updated_before[result_test_updated_before[:, 1] > 0] = 4

        ################################################################
        #                           Original                           #
        ################################################################

        box("Original")
        print()
        print("Timeframe from {} to {}".format(
            str(datetime.fromtimestamp(y_time[mask_test].min())),
            str(datetime.fromtimestamp(y_time[mask_test].max())),
        ))

        header("Statistics - Workload Reduction")
        datapoints = y_pred_test_orig.shape[0]
        automated  = (y_pred_test_orig >= 0).sum()
        anomalous  = (y_pred_test_orig <  0).sum()
        anomalies_conf  = (y_pred_test_orig == -1).sum()
        anomalies_train = (y_pred_test_orig == -2).sum()
        anomalies_eps   = (y_pred_test_orig == -3).sum()
        width = len(str(datapoints))
        print("Datapoints            : {:{width}}".format(datapoints, width=width))
        print("Automated             : {:{width}}/{:{width}} = {:6.2f}%".format(automated, datapoints, 100*automated/datapoints, width=width))
        print("Anomalous             : {:{width}}/{:{width}} = {:6.2f}%".format(anomalous, datapoints, 100*anomalous/datapoints, width=width))
        print("Anomalous confidence  : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_conf, datapoints, 100*anomalies_conf/datapoints, width=width))
        print("Anomalous not in train: {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_train, datapoints, 100*anomalies_train/datapoints, width=width))
        print("Anomalous > epsilon   : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_eps, datapoints, 100*anomalies_eps/datapoints, width=width))
        print()

        header("Statistics - Anomalies")
        y_true_anomalous = y_true_test[y_pred_test_orig < 0]
        y_pred_anomalous = y_pred_test_orig[y_pred_test_orig < 0]

        print(confusion_report(
            y_true = y_true_anomalous,
            y_pred = y_pred_anomalous,
            labels = [-3, -2, -1, 0, 1, 2, 3, 4],
            target_names  = ['CONF', 'TRAIN', 'EPS',
                             'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            skip_x = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            skip_y = ['CONF', 'TRAIN', 'EPS'],
        ))

        header("Performance")
        mask = y_pred_test_orig >= 0
        print(classification_report(
            y_true        = y_true_test[mask],
            y_pred        = y_pred_test_orig[mask],
            digits        = 4,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            zero_division = 0,
        ))

        header("Confusion matrix")
        print(confusion_report(
            y_true        = y_true_test[mask],
            y_pred        = y_pred_test_orig[mask],
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        ))

        ################################################################
        #                       Updated - After                        #
        ################################################################

        box("Updated - After")
        print()
        print("Timeframe from {} to {}".format(
            str(datetime.fromtimestamp(y_time[mask_test].min())),
            str(datetime.fromtimestamp(y_time[mask_test].max())),
        ))

        header("Statistics - Workload Reduction")
        datapoints = y_pred_test_updated.shape[0]
        automated  = (y_pred_test_updated >= 0).sum()
        anomalous  = (y_pred_test_updated <  0).sum()
        anomalies_conf  = (y_pred_test_updated == -1).sum()
        anomalies_train = (y_pred_test_updated == -2).sum()
        anomalies_eps   = (y_pred_test_updated == -3).sum()
        width = len(str(datapoints))
        print("Datapoints            : {:{width}}".format(datapoints, width=width))
        print("Automated             : {:{width}}/{:{width}} = {:6.2f}%".format(automated, datapoints, 100*automated/datapoints, width=width))
        print("Anomalous             : {:{width}}/{:{width}} = {:6.2f}%".format(anomalous, datapoints, 100*anomalous/datapoints, width=width))
        print("Anomalous confidence  : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_conf, datapoints, 100*anomalies_conf/datapoints, width=width))
        print("Anomalous not in train: {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_train, datapoints, 100*anomalies_train/datapoints, width=width))
        print("Anomalous > epsilon   : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_eps, datapoints, 100*anomalies_eps/datapoints, width=width))
        print()

        header("Statistics - Anomalies")
        y_true_anomalous = y_true_test[y_pred_test_updated < 0]
        y_pred_anomalous = y_pred_test_updated[y_pred_test_updated < 0]

        print(confusion_report(
            y_true = y_true_anomalous,
            y_pred = y_pred_anomalous,
            labels = [-3, -2, -1, 0, 1, 2, 3, 4],
            target_names  = ['CONF', 'TRAIN', 'EPS',
                             'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            skip_x = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            skip_y = ['CONF', 'TRAIN', 'EPS'],
        ))

        header("Performance")
        mask = y_pred_test_updated >= 0
        print(classification_report(
            y_true        = y_true_test[mask],
            y_pred        = y_pred_test_updated[mask],
            digits        = 4,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            zero_division = 0,
        ))

        header("Confusion matrix")
        print(confusion_report(
            y_true        = y_true_test[mask],
            y_pred        = y_pred_test_updated[mask],
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        ))

        ################################################################
        #                       Updated - Before                       #
        ################################################################

        box("Updated - Before")
        print()
        print("Timeframe from {} to {}".format(
            str(datetime.fromtimestamp(y_time[mask_test].min())),
            str(datetime.fromtimestamp(y_time[mask_test].max())),
        ))

        header("Statistics - Workload Reduction")
        datapoints = y_pred_test_updated_before.shape[0]
        automated  = (y_pred_test_updated_before >= 0).sum()
        anomalous  = (y_pred_test_updated_before <  0).sum()
        anomalies_conf  = (y_pred_test_updated_before == -1).sum()
        anomalies_train = (y_pred_test_updated_before == -2).sum()
        anomalies_eps   = (y_pred_test_updated_before == -3).sum()
        width = len(str(datapoints))
        print("Datapoints            : {:{width}}".format(datapoints, width=width))
        print("Automated             : {:{width}}/{:{width}} = {:6.2f}%".format(automated, datapoints, 100*automated/datapoints, width=width))
        print("Anomalous             : {:{width}}/{:{width}} = {:6.2f}%".format(anomalous, datapoints, 100*anomalous/datapoints, width=width))
        print("Anomalous confidence  : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_conf, datapoints, 100*anomalies_conf/datapoints, width=width))
        print("Anomalous not in train: {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_train, datapoints, 100*anomalies_train/datapoints, width=width))
        print("Anomalous > epsilon   : {:{width}}/{:{width}} = {:6.2f}%".format(anomalies_eps, datapoints, 100*anomalies_eps/datapoints, width=width))
        print()

        header("Statistics - Anomalies")
        y_true_anomalous = y_true_test[y_pred_test_updated_before < 0]
        y_pred_anomalous = y_pred_test_updated_before[y_pred_test_updated_before < 0]

        print(confusion_report(
            y_true = y_true_anomalous,
            y_pred = y_pred_anomalous,
            labels = [-3, -2, -1, 0, 1, 2, 3, 4],
            target_names  = ['CONF', 'TRAIN', 'EPS',
                             'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            skip_x = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            skip_y = ['CONF', 'TRAIN', 'EPS'],
        ))

        header("Performance")
        mask = y_pred_test_updated_before >= 0
        print(classification_report(
            y_true        = y_true_test[mask],
            y_pred        = y_pred_test_updated_before[mask],
            digits        = 4,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            zero_division = 0,
        ))

        header("Confusion matrix")
        print(confusion_report(
            y_true        = y_true_test[mask],
            y_pred        = y_pred_test_updated_before[mask],
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        ))

        # Go to next timeframe
        month    += 1
        time_test = time_test_end
