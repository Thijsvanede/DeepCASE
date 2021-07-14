import argparse
import csv
import json
import numpy as np
import warnings
from collections import Counter
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Count event per class")
    parser.add_argument("file", help="input file to analyze")
    parser.add_argument("decoding", nargs='?', help="if given, use file for decoding")
    parser.add_argument("-m", "--max", type=float, default=float('inf'), help="maximum number of items")
    args = parser.parse_args()

    # Get decoding
    try:
        with open(args.decoding or "{}.encoding.json".format(args.file)) as dec:
            decoding = {
                k: {i: value for i, value in enumerate(v)}
                for k, v in json.load(dec).items()
            }
    except:
        warnings.warn("No decoding found")
        decoding = {}

    # Initialise result
    result = list()

    with open(args.file) as infile:
        reader = csv.reader(infile)
        next(reader)
        for i, data in enumerate(tqdm(reader)):
            if i > args.max: break
            # Get breach, impact and threat
            threat = int(data[-3])
            impact = int(data[-2])
            breach = int(data[-1])
            # Add to result
            result.append([impact, breach, threat])

    # Get result as array
    result = np.asarray(result)
    unknown = result[result[:, 1] == -2][:, 2]
    info    = result[result[:, 0] == -1][:, 2]
    low     = result[np.logical_and(result[:, 1] == 0, result[:, 0] >=  0, result[:, 0] <  30)][:, 2]
    medium  = result[np.logical_and(result[:, 1] == 0, result[:, 0] >= 30, result[:, 0] <  70)][:, 2]
    high    = result[np.logical_and(result[:, 1] == 0, result[:, 0] >= 70)][:, 2]
    attack  = result[result[:, 1] >   0][:, 2]

    # Transform to counters
    unknown = Counter([decoding.get('threat_name', {}).get(x, x) for x in unknown])
    info    = Counter([decoding.get('threat_name', {}).get(x, x) for x in info   ])
    low     = Counter([decoding.get('threat_name', {}).get(x, x) for x in low    ])
    medium  = Counter([decoding.get('threat_name', {}).get(x, x) for x in medium ])
    high    = Counter([decoding.get('threat_name', {}).get(x, x) for x in high   ])
    attack  = Counter([decoding.get('threat_name', {}).get(x, x) for x in attack ])

    print("UNKNOWN")
    for k, v in sorted(unknown.items()):
        print("{:60}: {:8}".format(k, v))
    print()

    print("INFO")
    for k, v in sorted(info.items()):
        print("{:60}: {:8}".format(k, v))
    print()

    print("LOW")
    for k, v in sorted(low.items()):
        print("{:60}: {:8}".format(k, v))
    print()

    print("MEDIUM")
    for k, v in sorted(medium.items()):
        print("{:60}: {:8}".format(k, v))
    print()

    print("HIGH")
    for k, v in sorted(high.items()):
        print("{:60}: {:8}".format(k, v))
    print()

    print("ATTACK")
    for k, v in sorted(attack.items()):
        print("{:60}: {:8}".format(k, v))
    print()
