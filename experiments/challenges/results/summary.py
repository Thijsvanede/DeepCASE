import argparse
import numpy as np
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    result = dict()

    with open(args.file) as infile:
        # Set mode
        mode = None

        # Loop over lines
        for line in infile:
            # Check if Mode changed
            re_mode = re.search('┃\s+(.*)\s+┃', line)
            if re_mode:
                mode = re_mode.group(1).strip()
                if mode not in result:
                    result[mode] = {
                        'coverage' : list(),
                        'precision': list(),
                        'recall'   : list(),
                        'f1-score' : list(),
                        'accuracy' : list(),
                        'support'  : list(),
                    }

            # Get coverage
            if 'Automated' in line:
                a, b = line.split()[2].split('/')
                result[mode]['coverage'].append(int(a)/int(b))

            if 'weighted avg' in line:
                _, _, precision, recall, f1_score, support = line.split()
                result[mode]['precision'].append(float(precision))
                result[mode]['recall'   ].append(float(recall))
                result[mode]['f1-score' ].append(float(f1_score))
                result[mode]['support'  ].append(float(support))

            if 'accuracy' in line:
                accuracy = line.split()[1]
                result[mode]['accuracy' ].append(float(accuracy))

    days = int(args.file.split('_')[0])
    for mode, values in result.items():
        if 'Before' in mode:
            continue
        print("{} - coverage".format(mode))
        for i, coverage in enumerate(values['coverage']):
            print("({:3}, {:.4f})".format(days*(i+1), coverage))
        print()
        print("{} - f1-score".format(mode))
        for i, score in enumerate(values['f1-score']):
            print("({:3}, {:.4f})".format(days*(i+1), score))
        print()

    for mode, result in result.items():
        support   = np.asarray(result['support'])
        coverage  = np.asarray(result['coverage'])
        precision = np.asarray(result['precision'])
        recall    = np.asarray(result['recall'])
        f1_score  = np.asarray(result['f1-score'])
        accuracy  = np.asarray(result['accuracy'])

        coverage  = (coverage  * support).sum() / support.sum()
        precision = (precision * support).sum() / support.sum()
        recall    = (recall    * support).sum() / support.sum()
        f1_score  = (f1_score  * support).sum() / support.sum()
        # accuracy  = (accuracy  * support).sum() / support.sum()
        accuracy = 0

        print(support.sum())

        print("{:16} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
            mode, coverage, precision, recall, f1_score, accuracy
        ))
