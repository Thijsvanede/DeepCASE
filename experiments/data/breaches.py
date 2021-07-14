import argparse
import json
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Breach statistics")
    parser.add_argument('file', help='input file containing breaches')
    args = parser.parse_args()

    # Get unique breaches
    breaches = dict()

    # Read input file
    with open(args.file) as infile:
        # Loop over datapoints
        for data in infile:
            # Read as dictionary
            data = json.loads(data)
            # Retrieve ID
            id = data.get('breach_uuid')

            if id in breaches:
                if breaches[id] != data:
                    print(breaches[id])
                    print(data)
                    raise ValueError("Double match")
            else:
                breaches[id] = data

    # Print statistics
    print("Total breaches      : {}".format(len(breaches)))
    print("Total hosts affected: {}".format(sum(x.get('hosts_affected') for x in breaches.values())))
    print("\nPhases")
    phases = Counter([x.get('breach_phase') for x in breaches.values()])
    width = max(len(str(x)) for x in phases.keys())
    for phase, count in sorted(phases.items(), key=lambda x: -x[1]):
        print("{:{width}}: {}".format(str(phase), count, width=width))
