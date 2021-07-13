import csv
import datetime
import itertools
import json
import logging
import numpy as np
import torch
import warnings
from collections import deque
from tqdm import tqdm

NONE = '-1337'

# Set logger
logger = logging.getLogger(__name__)

class NewlineLoader(object):

    def load(self, infile, dim_in, dim_out=1, max=float('inf')):
        """Load sequences from input file

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            dim_out : int, default=1
                Dimension of output sequence

            max : float, default=inf
                Maximum number of events to extract
            """
        # Initialise result
        X = list()
        y = list()

        # Load all items through iterator
        for X_, y_ in tqdm(self._load_(infile, dim_in, dim_out, max)):
            X.append(X_)
            y.append(y_)

        # Return result as tensors
        return torch.as_tensor(X), torch.as_tensor(y)

    def _load_(self, infile, dim_in, dim_out=1, max=float('inf')):
        """Load sequences from input file
            Assumes each file to contain host, event tuples on each line,
            separated by space, sorted by time.

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            dim_out : int, default=1
                Dimension of output sequence

            max : float, default=inf
                Maximum number of events to extract
            """
        # Initialise sequences
        sequences = dict()

        # Open input file
        with open(infile) as infile:
            # Read events
            for i, event in enumerate(infile):
                # Break on max
                if i >= max: break
                # Split event into host and event
                host, event = map(int, event.split())
                # Append event to sequences
                sequence = sequences.get(host, deque())
                sequence.append(event)
                # Yield sequence if long enough
                if len(sequence) == dim_in + dim_out:
                    # Yield result
                    yield list(sequence)[:dim_in], list(sequence)[-dim_out:]
                    # Remove last element from sequence
                    sequence.popleft()

                # Store sequence
                sequences[host] = sequence

class SimpleLoader(object):

    def load(self, infile, dim_in, dim_out=1, max=float('inf')):
        """Load sequences from input file

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            dim_out : int, default=1
                Dimension of output sequence

            max : float, default=inf
                Maximum number of items to extract
            """
        # Initialise result
        X = list()
        y = list()

        # Load all items through iterator
        for X_, y_ in tqdm(self._load_(infile, dim_in, dim_out, max)):
            X.append(X_)
            y.append(y_)

        # Return result as tensors
        return torch.as_tensor(X), torch.as_tensor(y)


    def _load_(self, infile, dim_in, dim_out=1, max=float('inf')):
        """Load sequences from input file as iterator

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            dim_out : int, default=1
                Dimension of output sequence

            max : float, default=inf
                Maximum number of items to extract

            Yields
            ------
            X : list
                Input sequence

            y : list
                Output sequence
            """
        # Initialise counter
        counter = 0

        # Open input file
        with open(infile, 'r') as infile:
            # Read sequence
            for sequence in infile.readlines():
                # Get sequence as integers
                sequence = list(map(int, sequence.strip().split()))
                # Get sequences
                for i in range(len(sequence) - dim_in - dim_out):
                    # Compute X and y of each sequence
                    X = sequence[i       :i+dim_in        ]
                    y = sequence[i+dim_in:i+dim_in+dim_out]

                    # Append to result
                    yield X, y

                    # Increment counter
                    counter += 1
                    # Return on maximum number of items
                    if counter >= max: return


class PreprocessLoader(object):

    def __init__(self):
        """Load preprocessed data"""
        # Create loader object
        self.loader = Loader()
        # Create filter for preprocessed data
        self.filter = Filter()

    def load(self, infile, dim_in, time_in=float('inf'), dim_out=1,
        max_events=float('inf'), max_sequences=float('inf'), offset=0,
        extract=[], encode={}, train_ratio=0.5, random=False):
        """Load sequences from input file

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            time_in : float, default=float('inf')
                Time of input sequence
                Shortcut sequences if there are no elements within the given
                timeframe

            dim_out : int, default=1
                Dimension of output sequence

            max_events : float, default=inf
                Maximum number of events to extract

            max_sequences : float, default=inf
                Maximum number of sequences to extract

            offset : float, default=0
                Offset of items to skip before loading

            extract : list
                Fields to extract

            encode : set, default={}
                Set of fields to encode

            train_ratio : float, default=0.5
                Ratio to train with. If 0 <= train_ratio <= 1, train with ratio
                otherwise train with given number of samples as int.

            random : boolean, default=False
                Whether to split randomly
            """
        logger.info("load {}".format(infile))

        # Load data
        data, encodings = self.load_sequences(
            infile        = infile,
            dim_in        = dim_in,
            time_in       = time_in,
            dim_out       = dim_out,
            max_events    = max_events,
            max_sequences = max_sequences,
            offset        = offset,
            extract       = extract,
            encode        = encode
        )

        # Split data
        data = self.split_train_test(data, train_ratio, random)
        # Split data on input and output
        for column, splits in data.items():
            for split, tensor in splits.items():
                if column == 'host':
                    data[column][split] = {'X': tensor, 'y': tensor}
                else:
                    data[column][split] = dict(zip(('X', 'y'), torch.split(
                        tensor,
                        [tensor.shape[1]-dim_out, dim_out],
                        dim=1,
                    )))

        # Return result
        return data, encodings

    def load_sequences(self, infile, dim_in, time_in=float('inf'), dim_out=1,
        max_events=float('inf'), max_sequences=float('inf'), offset=0,
        extract=[], encode={}):
        """Load sequences from input file

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int, optional
                Dimension of input sequence

            time_in : float, default=float('inf')
                Time of input sequence
                Shortcut sequences if there are no elements within the given
                timeframe

            dim_out : int, default=1
                Dimension of output sequence

            max_events : float, default=inf
                Maximum number of events to extract

            max_sequences : float, default=inf
                Maximum number of sequences to extract

            offset : float, default=0
                Offset of items to skip before loading

            extract : list, default=[]
                Fields to extract

            encode : set, default={}
                Set of fields to encode

            Returns
            -------
            data : dict()
                Dictionary of key -> data

            encodings : dict()
                Dictionary of key -> mapping
            """
        logger.info("load_sequences")

        # Initialise encodings
        encodings = {k: {NONE: 0} for k in ['host'] + list(encode)}
        # Initialise output
        result    = {k: list() for k in ['host'] + extract}

        # Read data
        data = self.loader.load(infile,
            max     = max_events,
            offset  = offset,
            verbose = max_sequences == float('inf'),
        )

        # Get decoding
        try:
            with open("{}.encoding.json".format(infile)) as file:
                # Read decoding as json
                decoding = json.load(file)
                # Transform
                for k, v in decoding.items():
                    decoding[k] = {str(i): item for i, item in enumerate(v)}
        except FileNotFoundError as e:
            warnings.warn("Could not decode: '{}'".format(e))

        # Define flexible time - TODO remove
        def key_time(x):
            try:
                result = float(x.get('ts_start'))
            except ValueError:
                result = datetime.datetime.strptime(
                    x.get('ts_start'),
                    "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            return result

        # Set data generator
        generator = self.filter.timegrams(data, time_in, dim_in+dim_out,
            max      = max_sequences,
            group    = lambda x:      (x.get('source'), x.get('src_ip')),
            key      = lambda x: tuple(x.get(item) for item in extract) ,
            key_time = key_time,
            decoding = decoding,
            verbose  = max_sequences != float('inf')
        )

        # Read sequences from data
        for host, sequence in generator:

            # Set host
            if host not in encodings['host']:
                encodings['host'][host] = len(encodings['host'])
            hosts = result.get('host', list())
            hosts.append(encodings['host'][host])
            result['host'] = hosts

            # Extract data
            for k, v in zip(extract, zip(*sequence)):
                # Transform data if necessary
                if k in encode:
                    v_ = list()
                    for x in v:
                        # Check if an encoding is available
                        if x not in encodings[k]:
                            # Update encoding
                            encodings[k][x] = len(encodings[k])
                        # Set encoded value
                        v_.append(encodings[k][x])
                    v = v_

                elif k == 'ts_start':
                    v = [int(float(x)) if x is not NONE else -1 for x in v]
                else:
                    try:
                        v = [int(x) if x is not NONE else -1 for x in v]
                    except TypeError:
                        pass

                # Update datapoint
                data_ = result.get(k, list())
                data_.append(v)
                result[k] = data_

        # Get data as tensors
        for k, v in result.items():
            logger.debug("Creating tensor for {}".format(k))
            if k == '_id':
                result[k] = np.asarray(v)
            else:
                try:
                    result[k] = torch.as_tensor(v).to(torch.int64)
                except Exception as e:
                    raise ValueError("Could not convert key '{}' to int tensor:"
                                     " '{}'".format(k, e))

        # Return result
        return result, encodings

    def split_train_test(self, data, train_ratio=0.5, random=False):
        """Split data in train and test sets

            Parameters
            ----------
            data : dict()
                Dictionary of identifier -> array-like of data

            train_ratio : float, default=0.5
                Ratio to train with. If 0 <= train_ratio <= 1, train with ratio
                otherwise train with given number of samples as int.

            random : boolean, default=False
                Whether to split randomly
            """
        logger.info("split_train_test")

        # Get number of samples
        n_samples = next(iter(data.values())).shape[0]

        # Select training and testing data
        # Initialise training
        i_train = np.zeros(n_samples, dtype=bool)
        if random:
            if 0 <= train_ratio <= 1:
                train_samples = round(train_ratio*i_train.shape[0])
            else:
                train_samples = round(train_ratio)

            # Set training data to randomly selected
            i_train[np.random.choice(
                    np.arange(i_train.shape[0]),
                    train_samples,
                    replace=False
            )] = True
        else:
            if 0 <= train_ratio <= 1:
                train_samples = round(n_samples*train_ratio)
            else:
                train_samples = round(train_ratio)

            # Set training data to first half
            i_train[:train_samples] = True
        # Testing is everything not in training
        i_test  = ~i_train

        # Split into train and test data
        for k, v in data.items():
            data[k] = {'train': v[i_train], 'test': v[i_test]}

        # Return result
        return data



################################################################################
#                    Object for filtering and grouping data                    #
################################################################################
class Filter(object):
    """Filter object for filtering and grouping json data."""

    def __init__(self):
        """Filter object for filtering and grouping json data."""
        pass

    def groupby(self, data, key):
        """Split data by key

            Parameters
            ----------
            data : iterable
                Iterable to split

            key : func
                Function by which to split data

            Yields
            ------
            key : Object
                Key value of item

            item : Object
                Datapoint of data
            """
        for k, v in itertools.groupby(data, key=key):
            for x in v:
                yield k, x

    def aggregate(self, data, group, key=lambda x: x):
        """Aggregate data by key

            Parameters
            ----------
            data : iterable
                Iterable to aggregate

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            Returns
            -------
            result : dict()
                Dictionary of key -> list of datapoints
            """
        # Initialise result
        result = dict()

        # Loop over datapoints split by key
        for k, v in self.groupby(data, group):
            # Add datapoint
            buffer = result.get(k, [])
            buffer.append(key(v))
            result[k] = buffer

        # Return result
        return result

    def timegrams(self, data, time, n, group, max=float('inf'), key=lambda x: x,
                  key_time=lambda x: x.get('time'), decoding={}, verbose=False):
        """Aggregate data by key of length given by maximum time difference

            Parameters
            ----------
            data : iterable
                Iterable to aggregate

            time : float
                Number of seconds events can occur before given gram

            n : int
                Length of n-gram

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            key_time : func, default=x.get('time')
                Function by which to extract time of datapoint

            verbose : boolean, default=False
                If True, print progress

            Yields
            ------
            k : object
                Key of given ngram

            ngram : tuplepass
                Tuple of datapoints for given ngram"""
        # Initialise result
        result = dict()
        times  = dict()

        # Keep track of yielded items
        yielded = 0
        # Keep track of progress
        if verbose:
            progress = tqdm(None,
                desc        = "Loading",
                total       = None if max == float('inf') else max,
                mininterval = 0,
                miniters    = 100,
            )

        for v in data:
            k = group(v)
        # # Loop over datapoints split by key
        # for k, v in self.groupby(data, group):
            # Add datapoint
            buffer_data = result.get(k, deque())
            buffer_time = times .get(k, deque())
            buffer_data.append(key     (v))
            buffer_time.append(key_time(v))

            # Yield if we find n-gram
            if len(buffer_data) >= n:
                # Yield buffer
                yielded += 1
                if verbose: progress.update()
                yield k, tuple(buffer_data)
                # Remove last item
                buffer_data.popleft()
                buffer_time.popleft()
            # Yield if we run out of time
            elif buffer_time[-1] - buffer_time[0] >= time:
                # Get buffer
                buffer = list(buffer_data)[1:]
                # Prefill buffer with None items
                pbuffer = [tuple([NONE]*len(buffer[-1]))]*(n-len(buffer))
                buffer  = pbuffer + buffer
                # Yield buffer
                yielded += 1
                if verbose: progress.update()
                yield k, tuple(buffer)
                # Remove last item
                buffer_data.popleft()
                buffer_time.popleft()

            if yielded >= max:
                if verbose: progress.close()
                return

            # Store buffer
            result[k] = buffer_data
            times [k] = buffer_time

    def ngrams(self, data, n, group, key=lambda x: x):
        """Aggregate data by key

            Parameters
            ----------
            data : iterable
                Iterable to aggregate

            n : int
                Length of n-gram

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            Yields
            ------
            k : object
                Key of given ngram

            ngram : tuple
                Tuple of datapoints for given ngram
            """
        # Initialise result
        result = dict()

        # Loop over datapoints split by key
        for k, v in self.groupby(data, group):
            # Add datapoint
            buffer = result.get(k, deque())
            buffer.append(key(v))
            # Yield if we find n-gram
            if len(buffer) >= n:
                # Yield buffer
                yield k, tuple(buffer)
                # Remove last item
                buffer.popleft()
            # Store buffer
            result[k] = buffer

    def signatures(self, data):
        """Generate signatures for each host in data

            Parameters
            ----------
            data : dict()
                Dictionary of host -> sequence

            Returns
            -------
            signatures : dict()
                Dictionary of host -> signature
            """
        # Initialise signatures
        signatures = dict()

        # Loop over each host to generate a signature
        for host, sequence in data.items():
            # Generate signature
            signature = "Variable"
            # Set signature
            if len(sequence) == 1:
                signature = "Single           {}".format(sequence[0])
            elif len(set(sequence)) == 1:
                signature = "Single repeating {}".format(sequence[0])

            # Set signature
            signatures[host] = signature

        # Return signatures
        return signatures

    def variable_ngrams(self, data, n, group, key=lambda x: x):
        """Return n-grams in data of length n only for non-trivial hosts

            Parameters
            ----------
            data : data : iterable
                Iterable to aggregate

            n : int
                Length of n-gram

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            Yields
            -------
            host : tuple
                Host identifier

            sequence : list of length n
                List containing datapoints of n-gram
            """
        # Aggregate data
        data = list(data)
        data_ = self.aggregate(data, group=group, key=key)
        # Generate signatures
        signatures = self.signatures(data_)
        # Extract only the hosts for which the signature is variable
        hosts = {k for k, v in signatures.items() if "Variable" in v}

        # Extract n-grams
        for host, sequence in self.ngrams(data, n, group=group, key=key):
            # Check if host in variable hosts
            if host in hosts:
                yield host, sequence



################################################################################
#                       Object for loading ndjson files                        #
################################################################################
class Loader(object):
    """Loader for data from preprocessed files"""

    def load(self, infile, max=float('inf'), offset=0, verbose=True):
        """Load data from given input file

            Parameters
            ----------
            infile : string
                Path to input file from which to load data

            max : float, default='inf'
                Maximum number of events to load from input file

            offset : float, default=0
                Offset of items to skip before loading

            verbose : boolean, default=True
                If True, print progress
            """
        # Read input file
        with open(infile) as infile:

            # Read offset more quickly
            header = next(infile).strip().split(',')
            i = 0
            while i < offset:
                next(infile)
                i += 1

            # Create csv reader
            reader = csv.DictReader(infile, fieldnames=header)

            # Set progress bar
            if verbose:
                reader = tqdm(reader,
                    desc        = "Loading",
                    total       = None if max == float('inf') else max,
                    mininterval = 0,
                    miniters    = 10000,
                )

            # Read data
            for i, data in enumerate(reader):
                # Update progress bar before completion
                if verbose and i == max - 1: reader.miniters = 1
                # Break on max
                elif i >= max:
                    break

                # Yield data
                yield data

            # Close progress if necessary
            if verbose: reader.close()



################################################################################
#                       Object for loading ndjson files                        #
################################################################################
class NdJsonLoader(object):
    """Loader for quickly loading data from ndjson files"""

    def __init__(self):
        """Data loader for quickly loading data from ndjson files"""
        pass

    def ndjson(self, file, max=float('inf'), field=None, verbose=False):
        """Load ndjson file

            Parameters
            ----------
            file : string
                File from which to read

            max : float, default=float('inf')
                Maximum number of items to read

            field : string, optional
                If given, only return given field
            """
        with open(file) as file:
            # Get file size
            file.seek(0, 2)
            size = file.tell()
            file.seek(0, 0)
            # Initialise progress
            i = 0
            # Read first line
            line = file.readline()

            # Read lines
            while line:
                # Print progress if necessary
                if verbose and i == (i >> 12 << 12):
                    print("Reading .ndjson file progress: {:.2f}%".format(
                        100*file.tell()/size
                    ), end='\r')

                # Break on maximum number of items
                if i >= max: break
                # Retireve data
                data = json.loads(line)
                # Yield data
                yield data.get(field, data)
                # Go to next line
                i += 1
                line = file.readline()


            # Print progress if necessary
            if verbose:
                print("Reading .ndjson file progress: {:.2f}%".format(
                    100*file.tell()/size
                ))

    def ndjson_write(self, file, data):
        """Write ndjson file

            Parameters
            ----------
            file : string
                File to write data to

            data : iterable
                Iterable to write
            """
        with open(file, 'w') as file:
            for d in data:
                d = json.dumps(d)
                file.write(d+'\n')


class FastParser(object):

    def __init__(self):
        pass

    def process(self, infile, outfile):
        """Process infile and output to outfile"""
        # Set custom variables
        fields  = set(['ts_start', 'threat_name', 'source', 'src_ip', 'impact', 'breach'])
        mapping = {
            'ts_start': lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()
        }


        # Loop over data
        data = self.read(infile)
        # Parse data
        data = self.parse(data,
            fields  = fields,
            mapping = mapping,
        )

        # Write data
        self.write(data, outfile, sorted(fields))

    def read(self, infile):
        """Read data from infile"""
        with open(infile) as infile:
            # Read data
            for data in csv.DictReader(infile):
                # Yield data
                yield data

    def parse(self, data, fields=set(), mapping=dict()):
        """Custom parse method, set fields and mapping."""
        # Loop over data
        for item in data:
            # Apply parse function to every item
            yield {
                k: mapping.get(k, lambda x: x)(v)
                for k, v in item.items()
                if k in fields
            }

    def write(self, data, outfile, fieldnames):
        """Write data to outfile"""
        with open(outfile, 'w') as outfile:
            # Create writer
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            # Write header
            writer.writeheader()
            # Write data
            for item in tqdm(data):
                writer.writerow(item)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Parse csv files")
    parser.add_argument("input" , help="input  file to parse")
    parser.add_argument("output", help="output file to parse")
    args = parser.parse_args()

    parser = FastParser()
    parser.process(args.input, args.output)
