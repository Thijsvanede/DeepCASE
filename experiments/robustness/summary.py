import numpy as np
import re

from utils import cm_to_cp

with open('robustness.result') as infile:

    result = dict()

    timeframe = None
    ratio     = None

    # Parse file
    for line in infile.readlines():
        # Skip log messages
        if  'Lookup table'     in line or\
            'Predicting'       in line or\
            'Optimizing query' in line:
            continue

        # Set timeframe
        elif 'Timeframe' in line:
            timeframe = line.strip()

            if timeframe not in result[ratio]:
                result[ratio][timeframe] = dict()

        # Set ratio
        elif re.search('┃\s*(.*)\s*┃', line):
            ratio = re.search('┃\s*(.*)\s*┃', line).group(1).strip()
            ratio = float(ratio.split(' = ')[1])

            if ratio not in result:
                result[ratio] = dict()

        elif 'Automated' in line:
            automated, total = map(int, line.split()[2].split('/'))
            result[ratio][timeframe]['automated'] = automated
            result[ratio][timeframe]['total'    ] = total

        elif re.search('\s*(INFO|LOW|MEDIUM|HIGH|ATTACK)\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)', line):
            regex = re.search('\s*(INFO|LOW|MEDIUM|HIGH|ATTACK)\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)', line)
            risk   = regex.group(1)
            matrix = np.asarray(list(map(int, [regex.group(2), regex.group(3), regex.group(4), regex.group(5), regex.group(6)])))
            result[ratio][timeframe][risk] = matrix

        # elif 'accuracy' in line:
        #     _, accuracy, support = line.split()
        #     accuracy = float(accuracy)
        #     support  = int(support)
        #
        #     result[ratio][timeframe]['accuracy'] = accuracy
        #     result[ratio][timeframe]['support' ] = support
        #
        # elif 'weighted avg' in line:
        #     precision, recall, f1_score = map(float, line.split()[2:-1])
        #
        #     result[ratio][timeframe]['precision'] = precision
        #     result[ratio][timeframe]['recall'   ] = recall
        #     result[ratio][timeframe]['f1-score' ] = f1_score

    # Create summary
    for ratio, timeframes in result.items():
        total = dict()
        for timeframe, metrics in timeframes.items():
            # Add anomaly metrics
            total['automated'] = total.get('automated', 0) + metrics.get('automated')
            total['total'    ] = total.get('total'    , 0) + metrics.get('total'    )

            # Add performance metrics
            matrix = np.vstack((
                metrics.get('INFO'),
                metrics.get('LOW'),
                metrics.get('MEDIUM'),
                metrics.get('HIGH'),
                metrics.get('ATTACK'),
            ))

            total['matrix'] = total.get('matrix', np.zeros((5, 5), dtype=int)) + matrix

        print("Ratio    : {}".format(ratio))
        print("Anomalous: {:8}/{:8} = {:.2f}%".format(
            total.get('total', 0) - total.get('automated', 0),
            total.get('total', 0),
            100 * (total.get('total', 0) - total.get('automated', 0)) / total.get('total', 0),
        ))
        print(cm_to_cp(total['matrix'], target_names=['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK']))



    # Create latex summary
    for ratio, timeframes in result.items():
        total = dict()
        for timeframe, metrics in timeframes.items():
            # Add anomaly metrics
            total['automated'] = total.get('automated', 0) + metrics.get('automated')
            total['total'    ] = total.get('total'    , 0) + metrics.get('total'    )

            # Add performance metrics
            matrix = np.vstack((
                metrics.get('INFO'),
                metrics.get('LOW'),
                metrics.get('MEDIUM'),
                metrics.get('HIGH'),
                metrics.get('ATTACK'),
            ))

            total['matrix'] = total.get('matrix', np.zeros((5, 5), dtype=int)) + matrix

        # Get metrics
        ratio = 100 * ratio
        anomalous = 100 * (total.get('total', 0) - total.get('automated', 0)) / total.get('total', 0)

        # Get classification report
        cp = cm_to_cp(total['matrix'])

        for line in cp.split('\n'):
            if 'weighted avg' in line:
                precision, recall, f1_score = map(float, line.split()[2:-1])
            if 'accuracy' in line:
                accuracy = float(line.split()[1])

        precision = 100 * precision
        recall    = 100 * recall
        f1_score  = 100 * f1_score
        accuracy  = 100 * accuracy

        print("{:6.2f}\\% & {:6.2f}\\% & {:6.2f}\\% & {:6.2f}\\% & {:6.2f}\\% & {:6.2f}\\% \\\\".format(
            ratio, anomalous, precision, recall, f1_score, accuracy,
        ))
