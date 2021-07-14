from collections     import Counter
from datetime        import datetime, timedelta
from sklearn.metrics import classification_report, homogeneity_score
import argformat
import argparse
import logging
import math
import numpy as np
import scipy
import sys
import torch
sys.path.insert(1, '../../deepseq/')

from deepseq           import DeepSeq
from context_builder   import ContextBuilder
from interpreter       import Interpreter
from preprocessing     import PreprocessLoader
from utils             import box, confusion_report, header
from interpreter.utils import lookup_table

def n_choose_k(n, k):
    if k > n: return 0
    return math.factorial(n) / (math.factorial(n-k) * math.factorial(k))

def probability_suspicious(classes, k):
    # Compute total
    N = classes.sum(axis=1)

    top = np.zeros(N.shape[0])
    # Compute factorials
    for c in range(classes.shape[1]):
        top += scipy.special.comb(classes[:, c], k)

    # Set result
    result = top / scipy.special.comb(N, k)
    # Remove nan values
    result[np.isnan(result)] = 0
    # Return result
    return 1 - result

def probability_high_risk(classes, k):
    # Compute total
    N = classes.sum(axis=1)

    # Compute non highest class
    processed = np.zeros(classes.shape[0], dtype=bool)
    top       = np.zeros(classes.shape[0], dtype=float)
    for c in reversed(range(classes.shape[1])):
        # Get mask of relevant items
        mask = classes[:, c] > 0

        # Get relevant mask
        mask_ = np.logical_and(~processed, mask)
        # Compute top
        top[mask_] = classes[mask_, :c].sum(axis=1)

        # Set processed items
        processed[mask] = True

    result = scipy.special.comb(top, k) / scipy.special.comb(N, k)
    # Remove nan values
    result[np.isnan(result)] = 0

    # Return result
    return 1 - result

def probability_either(classes, k):
    # Compute total
    N = classes.sum(axis=1)

    top = np.zeros(N.shape[0])
    highest = np.zeros(classes.shape[0], dtype=bool)
    # Compute factorials
    for c in reversed(range(classes.shape[1])):
        # If label is highest
        top[highest] += scipy.special.comb(classes[highest, c], k)

        # Add highest as allowed
        highest[classes[:, c] > 0] = True

    # Set result
    result = top / scipy.special.comb(N, k)
    # Remove nan values
    result[np.isnan(result)] = 0
    # Return result
    return 1 - result


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

    # Load Interpreter if necessary
    if args.load_interpreter:
        deepseq.interpreter = Interpreter.load(
            infile          = args.load_interpreter,
            context_builder = deepseq.context_builder,
        )
    # Fit Interpreter
    else:
        score_y = torch.stack((y_impact_train, y_breach_train), dim=2)
        deepseq.interpreter.fit(X_train, y_train,
            score   = score_y.squeeze(1),
            verbose = args.verbose,
        )

    # Save Interpreter if necessary
    if args.save_interpreter:
        deepseq.interpreter.save(args.save_interpreter)

    scores = np.zeros((deepseq.interpreter.clusters.shape[0], 2))
    scores[deepseq.interpreter.clusters != -1] = deepseq.interpreter.scores

    clusters = {
        'info'  : list(),
        'low'   : list(),
        'medium': list(),
        'high'  : list(),
        'attack': list(),
        'mixed' : list(),
        'total' : list(),
    }

    adjacent = {
        'info'  : list(),
        'low'   : list(),
        'medium': list(),
        'high'  : list(),
    }

    matrix = list()

    classes = np.zeros(deepseq.interpreter.clusters.shape[0]) - 1

    # Loop over all clusters
    for cluster, indices in lookup_table(deepseq.interpreter.clusters):
        if cluster != -1:
            # Get corresponding scores
            scores_ = scores[indices]
            # Get type
            attack  = scores_[:, 1] > 0
            attack  = attack
            info    = np.logical_and(~attack, scores_[:, 0] <= 0)
            low     = np.logical_and(np.logical_and(~attack, scores_[:, 0] >   0), scores_[:, 0] < 30)
            medium  = np.logical_and(np.logical_and(~attack, scores_[:, 0] >= 30), scores_[:, 0] < 70)
            high    = np.logical_and(~attack, scores_[:, 0] >= 70)

            classes[indices[info  ]] = 0
            classes[indices[low   ]] = 1
            classes[indices[medium]] = 2
            classes[indices[high  ]] = 3
            classes[indices[attack]] = 4

            # Count
            info   = info  .sum()
            low    = low   .sum()
            medium = medium.sum()
            high   = high  .sum()
            attack = attack.sum()

            matrix.append([info, low, medium, high, attack])

            if info == indices.shape[0]:
                clusters['info'  ].append(indices)
            elif low == indices.shape[0]:
                clusters['low'   ].append(indices)
            elif medium == indices.shape[0]:
                clusters['medium'].append(indices)
            elif high == indices.shape[0]:
                clusters['high'  ].append(indices)
            elif attack == indices.shape[0]:
                clusters['attack'].append(indices)
            else:
                clusters['mixed' ].append(indices)

                if bool(info) + bool(low) + bool(medium) + bool(high) + bool(attack) == 2:
                    if info and low:
                        adjacent['info'  ].append(indices)
                    elif low and medium:
                        adjacent['low'   ].append(indices)
                    elif medium and high:
                        adjacent['medium'].append(indices)
                    elif high and attack:
                        adjacent['high'  ].append(indices)

            clusters['total'].append(indices)

    for k, v in clusters.items():
        print(k)
        print("Clusters: {}".format(len(v)))
        v = np.asarray([x.shape[0] for x in v])
        print("Avg    size: {}".format(v.mean()))
        print("Median size: {}".format(np.median(v)))
        print("Std    size: {}".format(v.std()))
        print("Min    size: {}".format(v.min()))
        print("Max    size: {}".format(v.max()))
        print("Total  size: {}".format(v.sum()))
        print()
        uniq, counts = np.unique(np.sort(v), return_counts=True)
        for u, c in zip(uniq, counts):
            if u > 50: break
            print('\t({}, {})'.format(u, c))
        print('\t(51, -1)')
        print('\t(57, {})'.format(counts[np.logical_and(uniq >=   50, uniq <   100)].sum()))
        print('\t(59, -1)')
        print('\t(62, {})'.format(counts[np.logical_and(uniq >=  100, uniq <  1000)].sum()))
        print('\t(64, -1)')
        print('\t(67, {})'.format(counts[np.logical_and(uniq >= 1000, uniq < 10000)].sum()))
        print('\t(69, -1)')
        print('\t(72, {})'.format(counts[uniq >= 10000].sum()))
        print('\t(74, -1)')

        # (51, -1)
        # (57, 132)
        # (59, -1)
        # (62, 197)
        # (64, -1)
        # (67, 46)
        # (69, -1)
        # (72, 16)
        # (74, -1)

    print("Adjacent clusters: {}".format(sum(len(v) for v in adjacent.values())))
    print("Adjacent size    : {}".format(sum(sum(x.shape[0] for x in v) for v in adjacent.values())))
    print()

    from sklearn.metrics import homogeneity_score
    print(homogeneity_score(classes, deepseq.interpreter.clusters))
    print(homogeneity_score(classes[deepseq.interpreter.clusters != -1], deepseq.interpreter.clusters[deepseq.interpreter.clusters != -1]))
    print()

    # Confidence
    matrix = np.asarray(matrix)
    matrix = matrix[(matrix > 0).sum(axis=1) > 1]

    print("SUSPICIOUS")
    for i in range(1, 201):
        avg_prob = probability_suspicious(matrix, i)
        print("({:2}, {:.4f})".format(i, np.average(avg_prob)))
    print()


    print("CONSERVATIVE")
    for i in range(1, 201):
        avg_prob = probability_high_risk(matrix, i)
        print("({:2}, {:.4f})".format(i, np.average(avg_prob)))
    print()

    print("EITHER")
    for i in range(1, 201):
        avg_prob = probability_either(matrix, i)
        weights = matrix.sum(axis=1)
        print("({:2}, {:.4f})".format(i, np.average(avg_prob)))
