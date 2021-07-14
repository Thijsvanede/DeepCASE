import argformat
import argparse
import json
import pandas as pd
import numpy  as np
from tqdm import tqdm

if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    parser = argparse.ArgumentParser(
        description     = "Compute #alerts triggered by expert rules",
        formatter_class = argformat.StructuredFormatter,
    )

    parser.add_argument('file', help=".ndjson file containing incidents")
    parser.add_argument('--csv', help=".csv file containing alerts")
    parser.add_argument('--offset'   , type=float, default=None, help="number of rows to skip - for train/test")

    args = parser.parse_args()

    ########################################################################
    #                         Compute rule matches                         #
    ########################################################################

    alerts = list()

    # Loading data
    with open(args.file) as infile:
        for line in tqdm(infile.readlines()):
            alerts.append(json.loads(line))

    # Compute number of alerts and number of events
    n_alerts = len(alerts)
    n_events = 0
    for alert in alerts:
        n_events += alert.get('events')

    total_events = 10_500_000
    total_events = 7_033_681
    width = len(str(total_events))

    print("Workload: {:{width}}/{:{width}} = {:8.4f}%".format(n_alerts, n_events    , 100*n_alerts/n_events    , width=width))
    print("Coverage: {:{width}}/{:{width}} = {:8.4f}%".format(n_events, total_events, 100*n_events/total_events, width=width))

    ########################################################################
    #                        Compute eveny matches                         #
    ########################################################################
    if args.offset:
        args.offset = range(1, int(args.offset))

    if args.csv:
        data = pd.read_csv(args.csv, skiprows=args.offset)
        n_events = np.logical_or(
            data['impact'] >= 0,
            data['breach'] >= 0,
        ).sum()

        total_events = data.shape[0]

        print()
        print("Workload: {:{width}}/{:{width}} = {:8.4f}%".format(n_alerts, n_events    , 100*n_alerts/n_events    , width=width))
        print("Coverage: {:{width}}/{:{width}} = {:8.4f}%".format(n_events, total_events, 100*n_events/total_events, width=width))

    # print(alerts[0])
    # print(alerts[5000])
    # print(alerts[-1])
