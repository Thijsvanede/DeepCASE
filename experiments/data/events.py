import argparse
import csv
import json
from collections import Counter
from tqdm        import tqdm

class Events(object):

    def __init__(self, infile, maximum=float('inf')):
        """Quick retrieval of events."""
        # Initialise events
        self.events = dict()

        # Get decoding
        with open("{}.encoding.json".format(infile)) as infile_encoding:
            # Read encoding
            self.encoding = json.load(infile_encoding)
            # Create decoding
            self.decoding = {k: {i: x for i, x in enumerate(v)} for k, v in encoding.items()}

        with open(infile) as infile:
            # Get csv reader
            reader = csv.DictReader(infile)
            # Loop over data
            for i, data in enumerate(tqdm(reader)):
                # Get data by key
                key, subkey, id = map(int, data.get('portal_url').split('/'))

                # Set data
                if key not in self.events:
                    self.events[key] = dict()
                if subkey not in self.events[key]:
                    self.events[key][subkey] = dict()
                if id in self.events[key][subkey]:
                    # Skip if datapoint is equal to match
                    if data == self.events[key][subkey][id]: continue

                    difference = dict()
                    for k, value in self.events[key][subkey][id].items():
                        if data[k] != value:
                            difference[k] = (value, data[k])

                    # Set to the highest impact/confidence
                    if set(difference.keys()) == set(['impact', 'confidence']):
                        impact     = max(list(map(int, difference['impact'])))
                        confidence = max(list(map(int, difference['confidence'])))
                        self.events[key][subkey][id]['impact']     = impact
                        self.events[key][subkey][id]['confidence'] = confidence
                        continue

                    # Set to the highest log level
                    if set(difference.keys()) == set(['operation']):
                        operation = max(list(map(int, difference['operation'])))
                        self.events[key][subkey][id]['operation'] = operation
                        continue

                    print()
                    data_ = dict()
                    for k, v in data.items():
                        if k in self.decoding:
                            data_[k] = self.decoding[k].get(int(v), v)
                        else:
                            data_[k] = v
                    print(data_)
                    print(difference)
                    print()
                else:
                    self.events[key][subkey][id] = data

                if i > maximum:
                    break

    ########################################################################
    #                               Iterator                               #
    ########################################################################


    ########################################################################
    #                               Get data                               #
    ########################################################################

    def get(self, key=None, subkey=None, id=None):
        """Get event by key, subkey or id."""
        # Initialise result
        result = list()

        # Key, subkey and id given
        if key is not None and subkey is not None and id is not None:
            # Return datapoint if it exists
            if self.events.get(key, {}).get(subkey, {}).get(id):
                return list([self.events
                                .get(key   , {})
                                .get(subkey, {})
                                .get(id)
                ])
            # Else return an empty list
            else:
                return list()

        # Key and subkey given
        elif key is not None and subkey is not None and id is None:
            return list(self.events.get(key, {}).get(subkey, {}).values())

        # Key given
        elif key is not None and subkey is None and id is None:
            for subkeys in self.events.get(key, {}).values():
                for subkey, ids in subkeys.items():
                    for id, data in ids.items():
                        result.append(data)

        # None given
        elif key is None and subkey is None and id is None:
            for key, subkeys in self.events.items():
                for subkey, ids in subkeys.items():
                    for id, data in ids.items():
                        result.append(data)

        else:
            raise NotImplementedError("Not implemented key={}, subkey={}, id={}"
                .format(
                'None' if key    is None else '<key>',
                'None' if subkey is None else '<subkey>',
                'None' if id     is None else '<id>',
            ))

        return result

    ########################################################################
    #                             Extract keys                             #
    ########################################################################

    def keys(self):
        """Returns event keys"""
        return self.events.keys()

    def subkeys(self, key=None):
        """Returns event subkeys"""
        # Initialise result
        result = set()

        # Return all subkeys if no key is given
        if key is None:
            # Loop over keys
            for key, subkeys in self.events.items():
                # Add all subkeys per key
                result |= set(subkeys.keys())

        # Return subkeys for specific key
        else:
            result = self.events.get(key, {}).keys()

        # Return result
        return result

    def ids(self, key=None, subkey=None):
        """Returns event ids"""
        # Initialise result
        result = set()

        # Return all ids if no key or subkey is given
        if key is None and subkey is None:
            # Loop over keys
            for key, subkeys in self.events.items():
                # Loop over subkeys
                for subkey, ids in subkeys.items():
                    # Add ids
                    result |= set(ids.keys())

        # Return all ids if only key is given
        elif key is not None and subkey is None:
            # Loop over subkeys for given key
            for subkey, ids in self.events.get(key, {}).items():
                # Add ids
                result |= set(ids.keys())

        # Return all ids if both key and subkey are given
        elif key is not None and subkey is not None:
            # Add ids
            result |= set(self.events.get(key, {}).get(subkey, {}).keys())

        # Return all ids if only subkey is given
        elif key is None and subkey is not None:
            # Loop over keys
            for key, subkeys in self.events.items():
                # Check if subkey is in key
                if subkey in subkeys:
                    result = set(subkeys.get(subkey).keys())

        else:
            raise ValueError("This should never happen")


        # Return result
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Event statistics")
    parser.add_argument('file', help='input file containing events')
    args = parser.parse_args()

    # Get decoding
    with open("{}.encoding.json".format(args.file)) as infile:
        # Read encoding
        encoding = json.load(infile)
        # Create decoding
        decoding = {k:{i:x for i,x in enumerate(v)} for k,v in encoding.items()}

    # events = Events(args.file)
    # for key in sorted(events.keys()):
    #     for subkey in sorted(events.subkeys(key)):
    #         for id in sorted(events.ids(key=key, subkey=subkey)):
    #             print(key, subkey, id)
    #             print("Get all            :", len(events.get()))
    #             print("Get key            :", len(events.get(key=key)))
    #             print("Get key, subkey    :", len(events.get(key=key, subkey=subkey)))
    #             print("Get key, subkey, id:", len(events.get(key=key, subkey=subkey, id=id)))
    #             print("Get key, subkey, id:", len(events.get(key=key, subkey=subkey, id=id+1)))
    #             break
    #         break
    #     break
    # exit()

    threats_log  = dict()
    threats_info = dict()

    # Read input file
    with open(args.file) as infile:
        reader = csv.DictReader(infile)
        # Loop over datapoints
        for data in tqdm(reader):
            # Get threat and threat level
            threat = decoding.get('threat_name').get(int(data.get('threat_name')))
            level  = decoding.get('operation'  ).get(int(data.get('operation'  )))

            # Increment corresponding counter
            counter = threats_log if level == "LOG" else threats_info
            counter[threat] = counter.get(threat, 0) + 1

    # Print statistics
    detectors = set(encoding.get('detector_name', [])) - set(["None"])
    threats   = encoding.get('threat_name')
    print("Detectors: {}".format(len(detectors)))
    print("Threats  : {}".format(len(threats)))

    print("\nLOGS")
    width = max(len(str(x)) for x in threats_log.keys())
    print("─"*(width+10))
    for threat, count in sorted(threats_log.items(), key=lambda x: x[0]):
        print("{:{width}}: {:8}".format(threat, count, width=width))
    print("─"*(width+10))
    print("{:{width}}: {:8}".format("Total", sum(threats_log.values()), width=width))

    print("\nINFO")
    print("─"*(width+10))
    width = max(len(str(x)) for x in threats_info.keys())
    for threat, count in sorted(threats_info.items(), key=lambda x: x[0]):
        print("{:{width}}: {:8}".format(threat, count, width=width))
    print("─"*(width+10))
    print("{:{width}}: {:8}".format("Total", sum(threats_info.values()), width=width))
