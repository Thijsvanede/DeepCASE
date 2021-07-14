import argformat
import argparse
import numpy  as np
import pandas as pd
from collections     import Counter
from deepcase.utils  import confusion_report
from sklearn.metrics import classification_report
from tqdm            import tqdm

if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    parser = argparse.ArgumentParser(
        description     = "Compute #alerts when throttled",
        formatter_class = argformat.StructuredFormatter,
    )

    parser.add_argument('file', help=".csv file containing alerts")
    parser.add_argument('--thresholds', nargs='+', type=int, default=[
              60, #  1 minute
            5*60, #  5 minutes
           15*60, # 15 minutes
           30*60, # 30 minutes
           60*60, #  1 hour
        24*60*60, #  1 day
    ], help='thresholds (in seconds) against which to test.')
    parser.add_argument('-m', '--max'   , type=float, help="max rows to iterate")
    parser.add_argument('--offset', type=float, help="skip OFFSET rows")

    args = parser.parse_args()

    # Set offset
    if args.offset:
        args.offset = range(1, int(args.offset))
    if args.max:
        args.max = int(args.max)

    ########################################################################
    #                          Compute throttling                          #
    ########################################################################

    # Loading data
    data = pd.read_csv(args.file, nrows=args.max, skiprows=args.offset)

    # Compute score
    score = np.zeros(data.shape[0], dtype=int)
    score[data['impact'] >   0] = 1
    score[data['impact'] >= 30] = 2
    score[data['impact'] >= 70] = 3
    score[data['breach'] >   0] = 4
    data['score'  ] = score

    # Sort by time
    data = data.sort_values(by='ts_start')

    # Set thresholds
    for threshold in args.thresholds:

        # Set prediction
        predict_common = np.full(data.shape[0], -1, dtype=int)
        predict_max    = np.full(data.shape[0], -1, dtype=int)

        # Initialise alerts after throttling
        throttle_global  = 0 # Only makes sense for cross-company attacks
        throttle_company = 0 # Can be realistic in some scenario's
        throttle_local   = 0 # Same as usecase for DeepCASE

        # Initialise alerts
        alert_indices = list()

        # Group by event - throttle_global
        for event, events in tqdm(data.groupby('threat_name'), desc='Threshold = {:{width}} sec'.format(threshold, width=len(str(max(args.thresholds))))):
            # Set time of last alert
            last_alert = 0

            # Loop over all times
            for time in events['ts_start'].values:

                # Check for first event
                if time - last_alert >= threshold:
                    # Raise alert
                    throttle_global += 1
                    # Don't raise alert for another 'threshold' seconds
                    last_alert = time

            # Group by company - throttle_company
            for organization, events in events.groupby('source'):
                # Set time of last alert
                last_alert = 0

                # Loop over all times
                for time in events['ts_start'].values:

                    # Check for first event
                    if time - last_alert >= threshold:
                        # Raise alert
                        throttle_company += 1
                        # Don't raise alert for another 'threshold' seconds
                        last_alert = time

                # Group by machine - throttle_local
                for machine, events in events.groupby('src_ip'):
                    # Set time of last alert
                    last_alert = 0

                    # Loop over all times and scores
                    for index, time, in zip(
                            events.index.values,
                            events['ts_start'].values,
                        ):

                        # Check for first event
                        if time - last_alert >= threshold:
                            # Compute prediction
                            if alert_indices:
                                alerts = Counter(data['score'].values[alert_indices])
                                predict_common[alert_indices] = alerts.most_common(1)[0][0]
                                predict_max   [alert_indices] = max(alerts.keys())

                            # Raise alert
                            throttle_local += 1
                            # Don't raise alert for another 'threshold' seconds
                            last_alert = time

                            # Empty alerts
                            alert_indices = list()

                        # Add indices
                        alert_indices.append(index)


                    # Compute prediction
                    if alert_indices:
                        alerts = Counter(data['score'].values[alert_indices])
                        predict_common[alert_indices] = alerts.most_common(1)[0][0]
                        predict_max   [alert_indices] = max(alerts.keys())
                        # Empty alerts
                        alert_indices = list()
                        alerts        = Counter()


        # for alert, events in tqdm(data.groupby('alert'), desc='Predicting'):
        #     predict_common[events.index] = events['score'].mode().values[0] # Most common
        #     predict_max   [events.index] = events['score'].max()            # Max
        data['predict_max']    = predict_max
        data['predict_common'] = predict_common


        print("Threshold: {} seconds".format(threshold))
        print("-"*40)
        print("Throttle global : {:{width}} / {:{width}} = {:8.4f}%".format(throttle_global , data.shape[0], 100*throttle_global /data.shape[0], width=len(str(data.shape[0]))))
        print("Throttle company: {:{width}} / {:{width}} = {:8.4f}%".format(throttle_company, data.shape[0], 100*throttle_company/data.shape[0], width=len(str(data.shape[0]))))
        print("Throttle local  : {:{width}} / {:{width}} = {:8.4f}%".format(throttle_local  , data.shape[0], 100*throttle_local  /data.shape[0], width=len(str(data.shape[0]))))
        print()
        print("Prediction common:")
        print(classification_report(
            y_true        = data['score'         ].values,
            y_pred        = data['predict_common'].values,
            digits        = 4,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            zero_division = 0,
        ))
        print()
        print(confusion_report(
            y_true        = data['score'         ].values,
            y_pred        = data['predict_common'].values,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        ))
        print()
        print("Prediction max:")
        print(classification_report(
            y_true        = data['score'      ].values,
            y_pred        = data['predict_max'].values,
            digits        = 4,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            zero_division = 0,
        ))
        print()
        print(confusion_report(
            y_true        = data['score'      ].values,
            y_pred        = data['predict_max'].values,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        ))
        print()
