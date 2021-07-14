import argparse
import json
import numpy as np
from collections import Counter
from tqdm import tqdm

def diff_dicts(a, b, missing=KeyError):
    """
    Find keys and values which differ from `a` to `b` as a dict.

    If a value differs from `a` to `b` then the value in the returned dict will
    be: `(a_value, b_value)`. If either is missing then the token from
    `missing` will be used instead.

    :param a: The from dict
    :param b: The to dict
    :param missing: A token used to indicate the dict did not include this key
    :return: A dict of keys to tuples with the matching value from a and b
    """
    return {
        key: (a.get(key, missing), b.get(key, missing))
        for key in dict(
            set(a.items()) ^ set(b.items())
        ).keys()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Incident statistics")
    parser.add_argument('file', help='input file containing incidents')
    args = parser.parse_args()

    # Get unique incidents
    incidents = dict()

    # Read input file
    with open(args.file) as infile:
        # Loop over datapoints
        for data in tqdm(infile):
            # Read as dictionary
            data = json.loads(data)
            # Retrieve ID
            id = (
                data.get('access_key_id'),
                data.get('subkey_id'),
                data.get('incident_id')
            )

            if id in incidents:
                difference = diff_dicts(data, incidents[id])
                if 'events' in difference:
                    # Set to maximum number of events
                    incidents[id]['events'] = sum(difference['events'])
                    data         ['events'] = sum(difference['events'])
                    difference = diff_dicts(data, incidents[id])
                if 'src_hostname' in difference:
                    hostnames = set(filter(None, difference.get('src_hostname')))

                    if len(hostnames) == 1:
                        hostname = list(hostnames)[0]
                        incidents[id]['src_hostname'] = hostname
                        data         ['src_hostname'] = hostname
                        if 'src_label' in difference:
                            if hostname not in difference['src_label']:
                                raise ValueError("Different source label")
                            else:
                                incidents[id]['src_label'] = hostname
                                data         ['src_label'] = hostname
                        difference = diff_dicts(data, incidents[id])
                    else:
                        continue

                if difference:
                    raise ValueError("Double match")
            else:
                incidents[id] = data

    # Print statistics
    print("Total incidents     : {}".format(len(incidents)))
    print("Total related events: {}".format(sum(x.get('events') for x in incidents.values())))

    impact_breaches = np.asarray([x.get('impact') for x in incidents.values() if x.get("breach_uuid") != None])
    print(impact_breaches.min())
    print(impact_breaches.max())
    print(impact_breaches.mean())
    print(impact_breaches.std())
    width = len(str(sum(x.get('events') for x in incidents.values())))

    print("Breaches    : {:{width}} [{:{width}} events]".format(sum([x.get("breach_uuid") != None for x in incidents.values()]),
                                                sum([x.get("events") for x in incidents.values() if x.get("breach_uuid") != None]),
                                                width=width))
    print("Impact >= 70: {:{width}} [{:{width}} events]".format(sum([x.get("impact"     ) >= 70   for x in incidents.values()]),
                                                sum([x.get("events") for x in incidents.values() if x.get("impact") >= 70]),
                                                width=width))
    print("Impact >= 30: {:{width}} [{:{width}} events]".format(sum([x.get("impact"     ) >= 30   for x in incidents.values()]),
                                                sum([x.get("events") for x in incidents.values() if x.get("impact") >= 30]),
                                                width=width))
    print("Impact >=  0: {:{width}} [{:{width}} events]".format(sum([x.get("impact"     ) >= 0   for x in incidents.values()]),
                                                sum([x.get("events") for x in incidents.values() if x.get("impact") >= 0]),
                                                width=width))

    import matplotlib.pyplot as plt
    plt.hist(impact_breaches, bins=20)
    plt.show()
