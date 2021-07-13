import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as cm

def confusion_report(y_true, y_pred, labels=None, target_names=None,
                     sample_weight=None, normalize=None, skip_x=set(),
                     skip_y=set()):
    # Compute matrix
    matrix = cm(
        y_true        = y_true,
        y_pred        = y_pred,
        labels        = labels,
        sample_weight = sample_weight,
        normalize     = normalize
    )

    if target_names is not None:
        assert labels       is not None
        assert target_names is not None
        assert len(labels) == len(target_names)

        # Add labels to matrix
        matrix = np.concatenate(([target_names], matrix))
        matrix = np.concatenate(([["T\\P"] + target_names], matrix.T)).T

    # Compute width of rows
    width = np.vectorize(len)(matrix).max()

    # Transform to string
    result = ""
    mask_x = [i for i, x in enumerate(matrix[0   ]) if x not in skip_x]
    mask_y = [i for i, x in enumerate(matrix[:, 0]) if x not in skip_y]
    for row in matrix[mask_y]:
        result += "\t".join(
            "{:>{width}}".format(element, width=width)
            for element in row[mask_x]
        ) + '\n'

    return result


################################################################################
#                                 Draw output                                  #
################################################################################
def box(text, width=60):
    """Draw text in box."""
    lines = list()
    text  = text.split()
    line = ""
    for word in text:
        line_ = line + word + ' '
        if len(line_) > width-1:
            lines.append(line.strip())
            line = word + ' '
        else:
            line = line_
    if line:
        lines.append(line)

    # Print top of box
    print("┏" + "━"*(width-2) + "┓")
    for line in lines:
        print("┃{:^{width}}┃".format(line, width=width-2))
    print("┗" + "━"*(width-2) + "┛")

def header(text, width=60):
    """Draw header."""
    print("\033[1m" + text + "\033[0m")
    print("━"*width)

################################################################################
#                                   Reports                                    #
################################################################################

def print_report(report, decoding):
    """Print classification report including decoding."""
    # Print report
    print("{:45} | {:>7} {:>7} {:>7} {:>8}".format(
          "Class", "Prec", "Rec", "F-1", "Support"))
    print("-"*80)
    for k, v in sorted(report.items(), key=lambda x: (len(x[0]), x[0])):
        try:
            k = int(float(k))
            print("[{:>3}] {:39} | {:7.4f} {:7.4f} {:7.4f} {:8}".format(
                k, decoding.get(k, "?")[:39],
                v.get('precision'),
                v.get('recall'),
                v.get('f1-score'),
                v.get('support')))
        except ValueError:
            if k == "accuracy":
                print("-"*80)
                print("{:61}   {:7.4f}".format(k, v))
            else:
                print("{:45} | {:7.4f} {:7.4f} {:7.4f} {:8}".format(
                    k,
                    v.get('precision'),
                    v.get('recall'),
                    v.get('f1-score'),
                    v.get('support')))

def multiprediction_report(y_test, y_pred, decoding, y_train=None, min_train=0):
    """Print multiprediction classification report including decoding"""
    y_test = np.asarray(y_test).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Count items
    if y_train is not None:
        y_train = Counter(y_train.reshape(-1).tolist())
    else:
        y_train = dict()

    # Initialise predictions
    y_preds = [y_pred[:,:,0]]

    # Add predictions for each level
    for i in range(1, y_pred.shape[-1]):
        y_pred_ = y_preds[-1].copy()
        y_pred_[y_test == y_pred[:,:,i]] = y_test[y_test == y_pred[:,:,i]]
        y_preds.append(y_pred_)

    if y_test.shape[0] and y_pred.shape[0]:
        reports = [classification_report(
                    y_test.flatten(),
                    y_pred.flatten(),
                    output_dict=True,
                    zero_division=0
                   ) for y_pred in y_preds]
    else:
        reports = [{
            -1            : {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            'accuracy'    : 0,
            'macro avg'   : {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
        }]
        decoding[-1] = "NO TRAINING SAMPLES FOUND"

    # Get all keys
    keys = set()
    for r in reports:
        for key in r.keys():
            try:    keys.add(int(key))
            except: pass

    # Print report heading
    print("{:45} | {}"
        .format("", ' | '.join("{:^47}"
        .format("In top " + str(i+1)) for i in range(len(reports)))))
    print("{:45} | {}"
        .format("Class", ' | '.join("{:>7} {:>7} {:>7} {:>8} {:>14}"
        .format("Prec", "Rec", "F-1", "Support", "Support train") for i in range(len(reports)))))
    print("-"*(45 + 50*len(reports)))

    # Compute realistic macro avg
    realistic = [{} for i in range(len(reports))]

    # Print classes
    for k in sorted(keys):
        # Print class name
        print("[{:>3}] {:39}".format(k, decoding.get(k, "?")[:39]), end="")
        # For each top value, print result
        for i, report in enumerate(reports):
            print(" | {:7.4f} {:7.4f} {:7.4f} {:8} {:>14}".format(
                report.get(str(k), {}).get('precision', 0),
                report.get(str(k), {}).get('recall', 0),
                report.get(str(k), {}).get('f1-score', 0),
                report.get(str(k), {}).get('support', 0),
                y_train.get(k, '?')
            ), end="")

            # Add report values for realistic macro classification
            if y_train.get(k, 0) >= min_train:
                p = realistic[i].get('precision', [])
                r = realistic[i].get('recall'   , [])
                f = realistic[i].get('f1-score' , [])

                p.append(report.get(str(k), {}).get('precision', 0))
                r.append(report.get(str(k), {}).get('recall'   , 0))
                f.append(report.get(str(k), {}).get('f1-score' , 0))

                realistic[i]['precision'] = p
                realistic[i]['recall'   ] = r
                realistic[i]['f1-score' ] = f
        print()

    # Print footer
    print("-"*(45 + 50*len(reports)))
    # Print accuracy
    print("{:45} | {}"  .format("accuracy", ' | '.join(
          "{:23.4f}{:24}".format(r.get('accuracy'), '') for r in reports)))
    # Print other metrics
    for k in ["macro avg", "weighted avg"]:
        print("{:45} | {}".format(k,
            ' | '.join("{:7.4f} {:7.4f} {:7.4f} {:8} {:>14}".format(
            r.get(k).get('precision'),
            r.get(k).get('recall'),
            r.get(k).get('f1-score'),
            r.get(k).get('support'),
            sum(y_train.values()) or '?')
            for r in reports)))

    # Print realistic macro avg
    print("{:45}".format("macro avg >= {} training sample(s)".format(min_train)), end="")
    for r in realistic:
        print(" | {:7.4f} {:7.4f} {:7.4f} {:23}".format(
            sum(r.get('precision', [0])) / len(r.get('precision', [0])),
            sum(r.get('recall'   , [0])) / len(r.get('recall'   , [0])),
            sum(r.get('f1-score' , [0])) / len(r.get('f1-score' , [0])),
            "",
        ), end="")
    print()

    # print("{:45} | {}".format("realistic macro avg",
    #     ' | '.join("{:7.4f} {:7.4f} {:7.4f} {:23}".format(
    #         sum(r.get('precision')) / len(r.get('precision')),
    #         sum(r.get('recall'   )) / len(r.get('recall'   )),
    #         sum(r.get('f1-score' )) / len(r.get('f1-score' ))
    #         for r in realistic
    #     ))))
