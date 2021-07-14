from collections     import Counter
from scipy.sparse    import csr_matrix, lil_matrix
from sklearn.cluster import DBSCAN
from time            import time
from tqdm            import tqdm
import argformat
import argparse
import numpy as np
import scipy as sp
import sys
import torch


from data                         import prepare_y
from deepcase.preprocessing_old   import PreprocessLoader, SimpleLoader, NONE
from deepcase.interpreter.cluster import Cluster2
from deepcase.interpreter.utils   import sp_unique
from deepcase.utils               import confusion_report

class NgramCluster(object):

    def __init__(self, n_classes, eps=0.1, min_samples=5):
        """Set parameters for clustering.

            Parameters
            ----------
            n_classes : int
                Maximum number of classes to expect

            eps : float, default=0.1
                Maximum distance between elements that can be in the same
                cluster.

            min_samples : int, default=5
                Minimum number of samples to store in database.
            """
        # Set number of classes
        self.n_classes = n_classes

        # Initialise parameters
        self.dbscan = DBSCAN(
            eps         = eps,
            min_samples = min_samples,
            metric      = 'minkowski',
            p           = 1,
        )

        self.cluster = Cluster2(p=1)

        # Set trained values
        self.X_train  = None
        self.W_train  = None
        self.clusters = None
        self.mapping  = None


    def preprocess(self, X, verbose=False):
        """Preprocess sequences like the DeepCASE ContextBuilder without attention.

            Parameters
            ----------
            X : array-like of shape=(n_samples_original, len_seq)
                Sequences to preprocess.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            unique : scipy.sparse.csr_matrix of shape=(n_samples_unique, n_classes)
                Unique preprocessed sequences as sparse crs matrix

            inverse : np.array of shape=(n_samples_original,)
                Inverse used to reconstruct original X.

            counts : np.array of shape=(n_samples_unique,)
                Number of occurences of each unique value.
            """
        # Print verbosity
        if verbose: print("Computing unique sequences")

        # Count unique only - Memory reduction and speedup
        unique, inverse, counts = torch.unique(
            torch.as_tensor(X),
            return_counts  = True,
            return_inverse = True,
            dim            = 0,
        )

        # Print verbosity
        if verbose: print("Unique sequences computed")

        # Transform to numpy arrays
        unique  = np.asarray(unique )
        inverse = np.asarray(inverse)
        counts  = np.asarray(counts )

        # Get original array dimensions
        if verbose: print("Started one-hot encoding")
        n_samples, seq_length = unique.shape
        # Flatten array
        unique = unique.reshape(-1)
        # One hot encode each array as sparse matrix - Memory reduction and speedup
        unique = csr_matrix(np.eye(self.n_classes))[unique]
        # Transform back to matrix
        unique = sp.sparse.vstack([
            csr_matrix(unique[start:start+seq_length].sum(axis=0))
            for start in range(0, unique.shape[0], seq_length)
        ], format='csr')
        if verbose: print("Finished one-hot encoding")

        # Get unique sparse values - Speedup
        unique, inverse_, count_ = sp_unique(unique)
        inverse = inverse_[inverse]
        _, counts = np.unique(inverse, return_counts=True)

        # Return result
        return unique, inverse, counts


    def fit(self, X, y, verbose=False):
        """Fit n-gram with sequences and corresponding outcomes.

            Parameters
            ----------
            X : array-like of shape=(n_samples, len_seq)
                Sequences to train with.

            y : array-like of shape=(n_samples,)
                Impact scores corresponding to each sequence.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            self : self
                Returns self.
            """
        # Preprocess values
        unique, inverse, counts = self.preprocess(X, verbose=verbose)

        # Create mapping for y
        mapping = dict()
        for y_, inverse_ in zip(np.asarray(y), inverse):
            if inverse_ not in mapping:
                mapping[inverse_] = Counter()
            mapping[inverse_].update([y_])
        mapping = np.asarray([v for k, v in sorted(mapping.items())])

        # Print verbosity
        if verbose: print("Clustering {} samples".format(unique.shape[0]))

        # Compute clusters
        start = time()
        clusters = self.dbscan.fit_predict(
            unique,
            sample_weight = counts,
        )
        # Print verbosity
        if verbose: print("Clustering took {:.4f} seconds".format(time() - start))

        # Only keep values that belong to a cluster
        self.X_train  = unique  [clusters != -1]
        self.W_train  = counts  [clusters != -1]
        self.clusters = clusters[clusters != -1]

        # Create mapping
        self.mapping = dict()
        for cluster, map in zip(self.clusters, mapping[clusters != -1]):
            if cluster not in self.mapping:
                self.mapping[cluster] = Counter()
            self.mapping[cluster] += map

        # Return self
        return self


    def predict(self, X, y=None, verbose=False):
        """Predict the impact of sequences in X according to related n-grams.

            Parameters
            ----------
            X : array-like of shape=(n_samples, len_seq)
                Sequences to predict.

            y : ignored
                Ignored.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Severity predictions or -1 if no prediction could be made.
            """
        # Preprocess values
        unique, inverse, counts = self.preprocess(X, verbose=verbose)

        # Combine with train data
        samples = sp.sparse.vstack((self.X_train, unique), format='csr')
        weights = np.concatenate((self.W_train, counts))

        # Print verbosity
        if verbose: print("Clustering {} samples".format(samples.shape[0]))

        # Compute clusters
        start = time()

        clusters = self.dbscan.fit_predict(
            samples,
            sample_weight = weights,
        )

        # Print verbosity
        if verbose: print("Clustering took {:.4f} seconds".format(time() - start))

        # Get mapping from original clusters to new clusters
        mapping = dict()
        for original, new in zip(self.clusters, clusters):
            # -1 should not be in original
            assert original != -1

            if original not in mapping:
                mapping[original] = new
            # Ensure no double mappings exist
            assert mapping[original] == new



        # Initialise result
        result = np.asarray([
            self.mapping.get(mapping.get(cluster, -1), Counter([-1]))
            for cluster in clusters[-unique.shape[0]:]
        ])

        # Choose strategy - Always choose maximum risk
        result_max = np.asarray([max(x.keys()) for x in result])
        # Choose strategy - Always choose most common value
        # In case of multiple most common values, choose the highest to avoid
        # underpredicting. This favours NgramCluster in the same way as DeepCASE
        result_common = np.asarray([max([k for k, v in x.items() if v == max(x.values())]) for x in result])
        # result_common = np.asarray([x.most_common(1)[0][0] for x in result]) # Not deterministic in case of equal count

        # Return result
        return result_max[inverse], result_common[inverse]


class NgramMatch(object):

    def __init__(self, min_samples=5):
        """Set parameters for matching.

            Parameters
            ----------
            min_samples : int, default=5
                Minimum number of samples to store in database.
            """
        self.min_samples = min_samples

        # Initialise lookup table of n-grams
        self.grams = dict()


    def fit(self, X, y, verbose=False):
        """Fit n-gram with sequences and corresponding outcomes.

            Parameters
            ----------
            X : array-like of shape=(n_samples, len_seq)
                Sequences to train with.

            y : array-like of shape=(n_samples,)
                Impact scores corresponding to each sequence.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            self : self
                Returns self.
            """
        # Cast to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Reset lookup table
        self.grams = dict()

        # Create iterator
        iterator = zip(X, y)
        # Add verbosity if necessary
        if verbose:
            iterator = tqdm(iterator, desc='Fit    ')

        # Store unique combinations
        for X_, y_ in iterator:
            # Transform to tuple
            X_ = tuple(X_)

            # Set first value to set
            if X_ not in self.grams:
                self.grams[X_] = Counter()
            # Add to grams
            self.grams[X_].update([y_])

        # Only keep the counts where total >= min_samples
        self.grams = {
            k: v for k, v in self.grams.items()
            if sum(v.values()) >= self.min_samples
        }

        # Return self
        return self


    def predict(self, X, y=None, verbose=False):
        """Predict the impact of sequences in X according to related n-grams.

            Parameters
            ----------
            X : array-like of shape=(n_samples, len_seq)
                Sequences to predict.

            y : ignored
                Ignored.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Severity predictions or -1 if no prediction could be made.
            """
        # Cast to numpy array
        X = np.asarray(X)

        # Initialise result
        result_common = np.full(X.shape[0], -1)
        result_max    = np.full(X.shape[0], -1)

        # Create iterator
        iterator = enumerate(X)
        # Add verbosity if necessary
        if verbose:
            iterator = tqdm(iterator, desc='Predict')

        # Loop over all elements to predict
        for i, X_ in iterator:
            # Transform to tuple for lookup
            X_ = tuple(X_)

            # If sequence is known, get corresponding prediction
            if X_ in self.grams:
                # Prediction is currently set to "Most common"
                result_common[i] = self.grams[X_].most_common(1)[0][0]
                # Use the line below for prediction "Maximum"
                result_max[i] = max(self.grams[X_].keys())

        # Return result
        return result_max, result_common


    def fit_predict(self, X, y, verbose=False):
        """Apply fit and predict sequentially.

            Parameters
            ----------
            X : array-like of shape=(n_samples, len_seq)
                Sequences to train and predict.

            y : array-like of shape=(n_samples,)
                Impact scores corresponding to each sequence.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Severity predictions or -1 if no prediction could be made.
            """
        # First apply fit
        return self.fit(
            X = X,
            y = y,
            verbose = verbose,

        # Then apply predict
        ).predict(
            X = X,
            y = y,
            verbose = verbose,
        )







if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog            = "deepseq.py",
        description     = "DeepSeq: providing meta-level contextual analysis of security alerts",
        formatter_class = argformat.StructuredFormatter,
    )

    # Input
    group_input = parser.add_argument_group("Input")
    group_input.add_argument('file'      ,                            help="read preprocessed input     file")
    group_input.add_argument('test'      , nargs='?' ,                help="read preprocessed test      file")
    group_input.add_argument('malicious' , nargs='?' ,                help="read preprocessed malicious file")
    group_input.add_argument('--offset'  , type=float, default=0    , help="offset for items to load")
    group_input.add_argument('--time'    , type=float, default=86400, help="max time length of input sequence")
    group_input.add_argument('--all'     , dest='all'     , action='store_true', help="perform experiment on all data")
    group_input.add_argument('--breach'  , dest='breach'  , action='store_true', help="read breaches")
    group_input.add_argument('--incident', dest='incident', action='store_true', help="read incidents")
    group_input.add_argument('--ignore'  , dest='ignore'  , action='store_true', help="ignore incident and breach info")

    # ContextBuilder parameters
    context_builder = parser.add_argument_group("ContextBuilder")
    context_builder.add_argument('-f', '--features'      , type=int  , default=280,          help="maximum number of expected features")
    context_builder.add_argument('-i', '--dim-input'     , type=int  , default=10,           help="length of input sequence")
    context_builder.add_argument('-o', '--dim-output'    , type=int  , default=1,            help="length of output sequence")
    context_builder.add_argument('-m', '--max-sequences' , type=float, default=float('inf'), help="maximum number of sequences ro read from input")
    context_builder.add_argument('-n', '--max-events'    , type=float, default=float('inf'), help="maximum number of events to read from input")
    context_builder.add_argument('-c', '--complexity'    , type=int  , default=128,          help="complexity of the model")

    # Training
    group_training = parser.add_argument_group("ContextBuilder training")
    group_training.add_argument('-b', '--batch-size', type=int, default=128   , help="batch size")
    group_training.add_argument('-d', '--device'    , type=str, default='auto', help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10    , help="number of epochs to train with")
    group_training.add_argument('-r', '--random'    , action='store_true'     , help="train with random selection")
    group_training.add_argument('-s', '--silent', dest='verbose', action='store_false', help="supress printing progress")
    group_training.add_argument('--train', type=float, default=0.5, help="training samples to use (or ratio of if 0 <= TRAIN <= 1)")

    # Interpreter parameters
    interpreter = parser.add_argument_group("Interpreter")
    interpreter.add_argument('--epsilon'    , type=float, default=0.1, help="interpreter epsilon     for clustering")
    interpreter.add_argument('--min-samples', type=int,   default=5,   help="interpreter min_samples for clustering")
    interpreter.add_argument('--threshold'  , type=float, default=0.2, help="interpreter confidence threshold for fingerprinting")

    # Store/load model
    group_model = parser.add_argument_group("Model I/O parameters")
    group_model.add_argument('--load-context'    , help="load context builder from LOAD file")
    group_model.add_argument('--load-interpreter', help="load interpreter     from LOAD file")
    group_model.add_argument('--save-context'    , help="save context builder to   SAVE file")
    group_model.add_argument('--save-interpreter', help="save interpreter     to   SAVE file")

    # Parse arguments
    args = parser.parse_args()

    # Set device if necessary
    if args.device is None or args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'










    if args.ignore:
        breach   = None
        incident = None
        extract  = ['threat_name']
    elif args.breach:
        breach   = 'impact_breach'
        incident = None
        extract = ['threat_name', breach]
    elif args.incident:
        breach   = None
        incident = 'incident_impact'
        extract  = ['threat_name', incident]
    else:
        breach   = 'breach'
        incident = 'impact'
        extract  = ['threat_name', incident, breach]

    # Create loader for preprocessed data
    loader = PreprocessLoader()

    # Load data
    data, encodings = loader.load(args.file, args.dim_input, args.time, args.dim_output,
        max_events    = args.max_events,
        max_sequences = args.max_sequences,
        offset        = args.offset,
        train_ratio   = args.train,
        random        = args.random,
        extract       = extract,
        # extract       = ['threat_name', '_id', 'severity', 'confidence', 'impact_incident', 'impact_breach', 'source'],
        encode        = {'threat_name', 'source'}
    )

    # Get short handles - data
    X_train = data.get('threat_name').get('train').get('X').to(args.device)
    y_train = data.get('threat_name').get('train').get('y').to(args.device)
    X_test  = data.get('threat_name').get('test' ).get('X').to(args.device)
    y_test  = data.get('threat_name').get('test' ).get('y').to(args.device)

    # Get short handles - breach
    if breach is not None:
        X_train_breach = data.get(breach).get('train').get('X')
        y_train_breach = data.get(breach).get('train').get('y')
        X_test_breach  = data.get(breach).get('test' ).get('X')
        y_test_breach  = data.get(breach).get('test' ).get('y')

        # Set known masks
        mask_known_train = ~((torch.cat((X_train_breach, y_train_breach), dim=1) == -2).any(dim=1))
        mask_known_test  = ~((torch.cat((X_test_breach , y_test_breach ), dim=1) == -2).any(dim=1))

        # Set negative breaches to 0
        X_train_breach[X_train_breach < 0] = 0
        y_train_breach[y_train_breach < 0] = 0
        X_test_breach [X_test_breach  < 0] = 0
        y_test_breach [y_test_breach  < 0] = 0
    else:
        X_train_breach = torch.zeros(X_train.shape, dtype=torch.long)
        y_train_breach = torch.zeros(y_train.shape, dtype=torch.long)
        X_test_breach  = torch.zeros(X_test.shape, dtype=torch.long)
        y_test_breach  = torch.zeros(y_test.shape, dtype=torch.long)

        # Set known masks
        mask_known_train = torch.ones(X_train.shape[0], dtype=torch.bool)
        mask_known_test  = torch.ones(X_test .shape[0], dtype=torch.bool)

    # Get short handles - incident
    if incident is not None:
        X_train_incident = data.get(incident).get('train').get('X')
        y_train_incident = data.get(incident).get('train').get('y')
        X_test_incident  = data.get(incident).get('test' ).get('X')
        y_test_incident  = data.get(incident).get('test' ).get('y')

        # Set negative incidents to 0
        X_train_incident[X_train_incident < 0] = 0
        y_train_incident[y_train_incident < 0] = 0
        X_test_incident [X_test_incident  < 0] = 0
        y_test_incident [y_test_incident  < 0] = 0
    else:
        X_train_incident = torch.zeros(X_train.shape, dtype=torch.long)
        y_train_incident = torch.zeros(y_train.shape, dtype=torch.long)
        X_test_incident  = torch.zeros(X_test.shape, dtype=torch.long)
        y_test_incident  = torch.zeros(y_test.shape, dtype=torch.long)

    # Remove data because it is no longer required
    del data

    # Prepare severity target
    y_train = prepare_y(y_train_incident, y_train_breach)
    y_test  = prepare_y(y_test_incident , y_test_breach )


    # ########################################################################
    # #                             N-gram Match                             #
    # ########################################################################
    #
    # ngram = NgramMatch(
    #     min_samples = 5,
    # )
    #
    # ngram.fit(X_train, y_train, verbose=True)
    # y_pred_max, y_pred_common = ngram.predict(X_test, verbose=True)
    #
    # ########################################################################
    # #                           Print prediction                           #
    # ########################################################################
    # from sklearn.metrics import classification_report
    #
    # print("NgramMatch")
    #
    # for y_pred, message in zip([y_pred_max, y_pred_common], ['max', 'common']):
    #
    #     print()
    #     print("Prediction risk: {}".format(message))
    #     print("-"*60)
    #
    #     print(classification_report(
    #         y_true        = y_test[y_pred != -1],
    #         y_pred        = y_pred[y_pred != -1],
    #         digits        = 4,
    #         labels        = [0, 1, 2, 3, 4],
    #         target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
    #         zero_division = 0,
    #     ))
    #
    #     print("-"*60)
    #
    #     print(confusion_report(
    #         y_true        = y_test,
    #         y_pred        = y_pred,
    #         labels        = [-1, 0, 1, 2, 3, 4],
    #         target_names  = ['-1', 'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
    #     ))
    #
    #     print("-"*60)
    #
    #     print("Unable to predict {}/{} = {:.2f}%".format(
    #         (y_pred == -1).sum(),
    #         y_pred.shape[0],
    #         100 * (y_pred == -1).sum() / y_pred.shape[0],
    #     ))
    #
    #     print("Number of n-grams stored: {}".format(len(ngram.grams)))
    #     print()
    #
    # exit()

    ########################################################################
    #                            N-gram Cluster                            #
    ########################################################################

    ngram = NgramCluster(
        n_classes   = 275,
        eps         = 0.1,
        min_samples = 5,
    )

    ngram.fit(X_train, y_train, verbose=True)
    y_pred_max, y_pred_common = ngram.predict(X_test, verbose=True)

    y_test = y_test.cpu().numpy()

    ########################################################################
    #                           Print prediction                           #
    ########################################################################
    from sklearn.metrics import classification_report

    print("NgramCluster")

    for y_pred, message in zip([y_pred_max, y_pred_common], ['max', 'common']):

        print(np.unique(y_pred, return_counts=True))
        print(np.unique(y_test, return_counts=True))
        print(np.unique(y_pred[y_pred != -1], return_counts=True))
        print(np.unique(y_test[y_pred != -1], return_counts=True))

        print()
        print("Prediction risk: {}".format(message))
        print("-"*60)

        print(classification_report(
            y_true        = y_test[y_pred != -1],
            y_pred        = y_pred[y_pred != -1],
            digits        = 4,
            labels        = [0, 1, 2, 3, 4],
            target_names  = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
            zero_division = 0,
        ))

        print("-"*60)

        print(confusion_report(
            y_true        = y_test,
            y_pred        = y_pred,
            labels        = [-1, 0, 1, 2, 3, 4],
            target_names  = ['-1', 'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        ))

        print(confusion_report(
            y_true        = y_test,
            y_pred        = y_pred,
            labels        = [-1, 0, 1, 2, 3, 4],
            target_names  = ['-1', 'INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK'],
        ))

        print("-"*60)

        print("Unable to predict {}/{} = {:.2f}%".format(
            (y_pred == -1).sum(),
            y_pred.shape[0],
            100 * (y_pred == -1).sum() / y_pred.shape[0],
        ))

        print("Number of clusters: {}".format(np.unique(ngram.clusters).shape[0]))
        print()
