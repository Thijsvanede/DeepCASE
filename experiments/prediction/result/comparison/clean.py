from argformat import StructuredFormatter
import argparse
import datetime
import numpy as np
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog        = "clean.py",
        description = "Clean data",
        formatter_class=StructuredFormatter
    )
    parser.add_argument('files', nargs='+', help='arguments to clean')
    args = parser.parse_args()

    # Generate output dictionary
    output = dict()

    # Loop over all input files
    for infile in args.files:
        # Extract parameters
        max    = re.search('max_(1?e?\d?)', infile).group(0).split('_')[1]
        max    = float(max) if max else None
        ratio  = float(re.search('ratio_(0\.\d)', infile).group(0).split('_')[1])
        random = 'random' in infile
        fold   = int(infile[-8])

        # Set current system we read out
        system     = None
        # Set current dictionary
        results = dict()
        # Set current output key
        key = (max, ratio, random)
        output[key] = output.get(key, dict())

        with open(infile) as infile:
            count = 0
            for line in infile:
                # Strip line
                line = line.strip()

                if line.startswith('[Epoch') and '100.00%' not in line:
                    continue

                # Extract system
                if 'prediction' in line:
                    system = line.split()[0]
                    results[system] = {
                        'time_train': list(),
                    }

                # Extract time
                if 'average loss = 0.0000' in line:
                    time_test = [float(x) for x in line.split()[-1].split(':')]
                    results[system]['time_test'] = datetime.timedelta(
                        hours   = int(time_test[0]),
                        minutes = int(time_test[1]),
                        seconds = int(time_test[2] // 1),
                        microseconds = int((time_test[2] - (time_test[2] // 1)) * 1000000)
                    )

                elif 'average loss' in line:
                    epoch_time = [float(x) for x in line.split()[-1].split(':')]
                    epoch_time = datetime.timedelta(
                        hours   = int(epoch_time[0]),
                        minutes = int(epoch_time[1]),
                        seconds = int(epoch_time[2] // 1),
                        microseconds = int((epoch_time[2] - (epoch_time[2] // 1)) * 1000000)
                    )
                    results[system]['time_train'].append(epoch_time)

                if 'accuracy' in line:
                    results[system]['accuracy'] = np.asarray([float(x.strip()) for x in line.split('|')[1:]])
                if 'weighted avg' in line:
                    metrics = np.asarray([[float(y) for y in x.split()] for x in line.split('|')[1:]])
                    results[system]['precision'    ] = metrics[:, 0]
                    results[system]['recall'       ] = metrics[:, 1]
                    results[system]['f1-score'     ] = metrics[:, 2]
                    results[system]['support'      ] = metrics[:, 3]
                    results[system]['support_train'] = metrics[:, 4]

        # Get average training time
        for system, result in results.items():
            avg_time = datetime.timedelta()
            for t in result.get('time_train'):
                avg_time += t
            results[system]['time_train'] = avg_time / len(result.get('time_train'))

        # Set output
        output[key][fold] = results

    # Initialise final result
    result = dict()
    n_folds = 0

    # Average output over all k-folds
    for key, folds in output.items():
        # Initialise average
        average = dict()

        for i, fold in folds.items():
            n_folds += 1
            for system, metrics in fold.items():
                for metric, result_ in metrics.items():
                    if 'time' in metric:
                        initial = datetime.timedelta()
                    else:
                        initial = np.zeros(result_.shape)

                    result[key] = result.get(key, {})
                    result[key][system] = result.get(key).get(system, {})
                    result[key][system][metric] = result.get(key, {})\
                                                        .get(system, {})\
                                                        .get(metric, initial) + result_

    # Compute average
    for key, systems in result.items():
        for system, metrics in systems.items():
            for metric, score in metrics.items():
                result[key][system][metric] = score / n_folds


    from pprint import pprint
    pprint(result)
