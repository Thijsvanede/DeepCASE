import logging
import numpy as np
import pickle
import scipy.sparse as sp
import torch
import warnings
from collections       import Counter
from sklearn.neighbors import KDTree
from tqdm              import tqdm

from .cluster import Cluster
from .utils   import lookup_table, unique_2d

# Set logger
logger = logging.getLogger(__name__)

class Interpreter(object):

    def __init__(self, context_builder, features, eps=0.1, min_samples=5,
                 threshold=0.2):
        """Interpreter for a given ContextBuilder.

            Parameters
            ----------
            context_builder : ContextBuilder
                ContextBuilder to interpret.

            features : int
                Number of different possible security events.

            eps : float, default=0.1
                Epsilon used for determining maximum distance between clusters.

            min_samples : int, default=5
                Minimum number of required samples per cluster.

            threshold : float, default=0.2
                Minimum required confidence in fingerprint before using it in
                training clusters.
            """
        logger.info("__init__(context_builder={}, features={}, eps={}, "
                    "min_samples={}, threshold={})".format(context_builder,
                    features, eps, min_samples, threshold))

        # Initialise ContextBuilder
        self.context_builder = context_builder

        # Initialise features
        self.features = features

        # Set parameters
        self.eps         = eps
        self.min_samples = min_samples
        self.threshold   = threshold

        # Objects
        self.tree     = dict()
        self.labels   = dict()
        self.clusters = np.zeros(0)
        self.scores   = np.zeros(0)

    ########################################################################
    #                         Fit/predict methods                          #
    ########################################################################

    def fit(self, X, y, score, iterations=100, batch_size=1024,
            func_score=lambda x: torch.mean(x, dim=0), verbose=False):
        """Fit the interpreter with samples.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context on which to train.

            y : torch.Tensor of shape=(n_samples, 1)
                Events on which to train.

            score : torch.Tensor of shape=(n_samples, n_scores?)
                Maliciousness score associated with each training sample.
                If the matrix has 2 dimensions, compute a score for each final
                dimension.

            iterations : int, default=100
                Number of iterations for query.

            batch_size : int, default=1024
                Size of batch for query.

            func_score : callable, default=lambda x: torch.mean(x, dim=0)
                Function to compute score per cluster.
                Default function computes the mean along the 0-axis.

            verbose : boolean, default=False
                If True, prints achieved speedup of clustering algorithm.

            Returns
            -------
            self : self
                Returns self.
            """
        logger.info("Fitting {} samples".format(X.shape[0]))

        # Compute clusters, fingerprints and attention
        self.clusters, fingerprints, attention = self.cluster(X, y,
            eps         = self.eps,
            min_samples = self.min_samples,
            threshold   = self.threshold,
            iterations  = iterations,
            batch_size  = batch_size,
            verbose     = verbose
        )

        mask = self.clusters != -1
        fingerprints = fingerprints[mask]
        attention    = attention   [mask].cpu()
        y            = y           [mask].cpu()
        score        = score       [mask].cpu()

        # Compute score per sample
        scores = score.to(torch.float)
        self.scores = scores.clone().cpu().numpy()

        # Average score per cluster
        for cluster, indices in lookup_table(self.clusters[mask], verbose=verbose):
            # Get score for each cluster
            score = scores[indices]

            # Set score of cluster to mean score
            scores[indices] = func_score(score)

        # Initialise tree
        self.tree   = dict()
        self.labels = dict()

        # Compute lookup table
        indices_y = lookup_table(y.squeeze(1),
            key     = lambda x: x.item(),
            verbose = verbose,
        )
        if verbose:
            indices_y = tqdm(indices_y, desc='Fitting interpreter')

        # Loop over events
        for y, indices in indices_y:
            logger.debug("KDTree for y={}, samples={}".format(y, indices.shape[0]))

            # Get unique fingerprints
            fps = lookup_table(fingerprints[indices].toarray(), hash=lambda x: x.data.tobytes())
            fps, fps_indices = zip(*fps)
            fps = np.asarray(fps)
            fps_indices = [torch.as_tensor(x) for x in fps_indices]

            # Set tree fingerprints
            self.tree[y] = KDTree(fps, p=1)

            # Set labels
            data, index, nodes, bounds = self.tree[y].get_arrays()
            assert np.all(data == fps)

            # Compute average score per cluster
            indices = torch.as_tensor(indices, dtype=torch.long)
            score = scores[indices]
            # Set score
            self.labels[y] = {
                i: score[m].mean(dim=0).cpu().tolist()
                for i, m in zip(index, fps_indices)
            }

        # Return self
        return self

    def predict(self, X, y, k=1, iterations=100, batch_size=1024,
                verbose=False):
        """Predict maliciousness of context samples.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context for which to predict maliciousness.

            y : torch.Tensor of shape=(n_samples, 1)
                Events for which to predict maliciousness.

            k : int, default=1
                Number of close clusters to consider.

            iterations : int, default=100
                Iterations used for optimization.

            batch_size : int, default=1024
                Batch size used for optimization.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Predicted maliciousness score.
                Positive scores are maliciousness scores.
                A score of 0 means we found a match that was not malicious.
                Special cases:

                * -1: Not confident enough for prediction
                * -2: Label not in training
                * -3: Closest cluster > epsilon
            """
        logger.info("predict {} samples".format(X.shape[0]))

        # Check value of k
        if k != 1:
            raise NotImplementedError("Prediction only implemented for k=1.")

        # Get unique samples
        X, y, inverse_result = unique_2d(X, y)

        # Initialise result
        result = np.full((X.shape[0], self.scores.shape[-1]), -1, dtype=float)

        # Compute fingerprints
        fingerprints, mask = self.fingerprints_optimized(X, y,
            threshold   = self.threshold,
            iterations  = iterations,
            batch_size  = batch_size,
            verbose     = verbose,
        )

        # Create looup table
        indices_y = lookup_table(y.squeeze(1),
            key     = lambda x: x.item(),
            verbose = verbose,
        )
        # Add progress if necessary
        if verbose: indices_y = tqdm(indices_y, desc='Predicting')

        for y, indices in indices_y:
            logger.debug("predict y={}".format(y))
            # Check if y is in training set
            if y not in self.tree:
                logger.debug("predict y={}: not in training set".format(y))
                result[indices] = -2
                continue

            # Get corresponding fingerprints
            fingerprints_ = fingerprints[indices]
            mask_         = mask        [indices].cpu().numpy()
            # Continue if there are no confident samples
            if not mask_.any(): continue

            # Get unique fingerprints
            fingerprints_, inverse = torch.unique(
                torch.Tensor(fingerprints_[mask_].toarray()),
                return_inverse = True,
                dim            = 0,
            )
            # Cast to numpy
            fingerprints_ = fingerprints_.cpu().numpy()
            inverse       = inverse      .cpu().numpy()

            k_ = min(self.tree[y].get_arrays()[1].shape[0], k)

            # Get closest cluster
            distance, neighbours = self.tree[y].query(fingerprints_,
                k               = k_,
                return_distance = True,
                dualtree        = fingerprints_.shape[0] >= 1e3,
            )
            # Distances should always be sorted
            assert (np.sort(distance, axis=1) == distance).all()
            # Get neighbour indices
            neighbours = self.tree[y].get_arrays()[1][neighbours]

            # Initialise result for given y
            result_ = np.full((fingerprints_.shape[0], self.scores.shape[-1]), -3, dtype=float)

            # Valid results are those closer than epsilon
            valid = distance[:, 0] <= self.eps
            logger.debug("predict y={}: {}/{} < epsilon".format(y, valid.sum(), valid.shape[0]))

            # Get valid neighbours
            neighbours = neighbours[:, 0][valid]
            if neighbours.shape[0] > 0:
                result_[valid] = [self.labels[y][n] for n in neighbours]

            # Set result
            result[indices[mask_]] = result_[inverse]

        # Return result
        return result[inverse_result.cpu().numpy()]

    def fit_predict(self, X, y, score, k=1, iterations=100, batch_size=1024,
                    verbose=False):
        """Call fit and predict method on same data in sequence.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context on which to train and predict.

            y : torch.Tensor of shape=(n_samples, 1)
                Events on which to train and predict.

            score : torch.Tensor of shape=(n_samples,)
                Maliciousness score associated with each training sample.

            k : int, default=1
                Number of close clusters to consider.

            iterations : int, default=100
                Number of iterations for query.

            batch_size : int, default=1024
                Size of batch for query.

            verbose : boolean, default=False
                If True, prints achieved speedup of clustering algorithm.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Predicted maliciousness score.
                Positive scores are maliciousness scores.
                Special cases:

                * -1: Not confident enough for prediction
                * -2: Label not in training
                * -3: Closest cluster > epsilon
            """
        logger.info("fit_predict {} samples".format(X.shape[0]))

        # Call fit and predict in sequence
        return self.fit(X, y,
            score      = score,
            iterations = iterations,
            batch_size = batch_size,
            verbose    = verbose,
        ).predict(X, y,
            k          = 1,
            iterations = 100,
            batch_size = 1024,
            verbose    = False,
        )

    ########################################################################
    #                        Fingerprint generation                        #
    ########################################################################

    def fingerprint(self, X, attention, n):
        """Compute contextual fingerprints for each input.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, sequence_length, input_dim)
                Context items from which to generate fingerprint.

            attention : torch.Tensor of shape=(n_samples, sequence_length)
                Attention for each input.

            n : int
                Number of possible events, determines the size of a fingerprint.

            Returns
            -------
            result : scipy.sparse.csc_matrix of shape=(n_samples, n)
                Sparse fingerprints of each context.
            """
        logger.info("fingerprint {} samples".format(X.shape[0]))

        # Initialise fingerprints
        fingerprints = sp.csc_matrix((X.shape[0], n))
        range        = np.arange( X.shape[0],     dtype = int  )

        # Set fingerprints
        for i, events in enumerate(torch.unbind(X, dim=1)):
            logger.debug("fingerprint dimension {}/{}".format(i+1, X.shape[1]))
            fingerprints += sp.csc_matrix(
                (attention[:, i].detach().cpu().numpy(),
                (range, events.cpu().numpy())),
                shape=(X.shape[0], n)
            )

        # Return fingerprints
        return fingerprints

    def fingerprints_optimized(self, X, y, threshold=0.2, iterations=100,
                               batch_size=1024, return_attention=False,
                               verbose=False):
        """Get optimal fingerprints of context after explaining it using query.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context to cluster.

            y : torch.Tensor of shape=(n_samples, 1)
                Events to cluster.

            threshold : float, default=0.2
                Minimum confidence required for fingerprinting.

            iterations : int, default=100
                Number of iterations for query.

            batch_size : int, default=1024
                Size of batch for query.

            return_attention : boolean, default=False
                If True, also returns attention

            verbose : boolean, default=False
                If True, prints achieved speedup of clustering algorithm.

            Returns
            -------
            fingerprints : scipy.sparse.lil_matrix of shape=(n_samples, dim_fingerprint)
                Sparse fingerprints for each input sample.
                Where mask == False, fingerprint = [0, 0, ..., 0]

            mask : np.array of shape=(n_samples,)
                Boolean array of masked fingerprints. True where input has
                confidence >= threshold, False otherwise.

            attention : np.array of shape=(n_samples, seq_length)
                Only returned if return_attention = True.
                Attention for each context item.
            """
        logger.info("fingerprints_optimized {} samples".format(X.shape[0]))

        ####################################################################
        #                        Optimize attention                        #
        ####################################################################

        # Get optimal confidence
        _, confidence, attention = self.explain(X, y,
            iterations = iterations,
            batch_size = batch_size,
            verbose    = verbose,
        )
        # Check where confidence is above threshold
        mask = (confidence >= threshold)

        logger.debug("fingerprints_optimized {}/{} > confidence".format(mask.sum(), mask.shape[0]))

        ####################################################################
        #                       Create fingerprints                        #
        ####################################################################

        fingerprints = self.fingerprint(X[mask], attention[mask], self.features)
        fingerprints = np.round(fingerprints, decimals=4).tolil()

        # Set fingerprints where mask == False
        logger.debug("fingerprints_optimized creating lil_matrix of shape=({}, {})".format(mask.shape[0], fingerprints.shape[1]))
        result = sp.lil_matrix(
            (mask.shape[0], fingerprints.shape[1]),
            dtype = float,
        )
        logger.debug("fingerprints_optimized inflating lil_matrix {}/{}".format(mask.sum(), fingerprints.shape[0]))
        result[mask.nonzero(as_tuple=True)[0].cpu().numpy()] = fingerprints

        logger.debug("fingerprints_optimized lil_matrix created")

        # Return result
        if return_attention:
            return result, mask, attention
        else:
            return result, mask

    ########################################################################
    #                          Context clustering                          #
    ########################################################################

    def cluster(self, X, y, eps=0.1, min_samples=5, threshold=0.2,
                iterations=100, batch_size=1024, verbose=False):
        """Cluster contexts in X for same output event y.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context to cluster.

            y : torch.Tensor of shape=(n_samples, 1)
                Events to cluster.

            eps : float, default=0.1
                Epsilon to use for clustering.

            min_samples : int, default=5
                Minimum number of samples per cluster.

            threshold : float, default=0.2
                Minimum confidence required for clustering.

            iterations : int, default=100
                Number of iterations for query.

            batch_size : int, default=1024
                Size of batch for query.

            verbose : boolean, default=False
                If True, prints achieved speedup of clustering algorithm.

            Returns
            -------
            clusters : np.array of shape=(n_samples,)
                Clusters per input sample.

            fingerprints : np.array of shape=(n_samples, dim_fingerprint)
                Fingerprints for each input sample.

            attention np.array of shape=(n_samples, seq_length)
                Attention for each input X.
            """
        logger.info("cluster {} samples".format(X.shape[0]))

        # Get fingerprints optimized
        fingerprints, mask, attention = self.fingerprints_optimized(X, y,
            threshold        = threshold,
            iterations       = iterations,
            batch_size       = batch_size,
            return_attention = True,
            verbose          = verbose,
        )

        # Apply mask
        X = X[mask]
        y = y[mask]

        logger.debug("cluster:initialising result shape={}".format(X.shape[0]))

        # Initialise result
        result = np.full(X.shape[0], -1, dtype=int)

        logger.debug("cluster:result initialised")

        ####################################################################
        #                       Create lookup table                        #
        ####################################################################

        indices_y = lookup_table(y.squeeze(1), key=lambda x: x.item(), verbose=verbose)
        if verbose: indices_y = tqdm(indices_y, desc="Clustering")

        ####################################################################
        #                          Cluster events                          #
        ####################################################################

        # Initialise clustering algorithm
        cluster = Cluster(p=1)

        # Loop over each event
        for event, context_mask in indices_y:
            logger.debug("cluster: y={} with {} samples".format(event, context_mask.shape[0]))

            # Compute clusters per event
            clusters = cluster.dbscan(
                X           = fingerprints[context_mask],
                eps         = eps,
                min_samples = min_samples,
                verbose     = False,
            )

            # Set clusters with unique identifiers
            offset = max(0, result.max() + 1)
            clusters[clusters != -1] += offset
            result[context_mask] = clusters

        # Set result where mask == -1
        result_= np.full(mask.shape[0], -1, dtype=int)
        result_[mask.cpu().numpy()] = result

        # Return result
        return result_, fingerprints, attention

    ########################################################################
    #              Explain events from their security context              #
    ########################################################################

    def explain(self, X, y, iterations=100, batch_size=1024, verbose=False):
        """Explain events y by computing optimal attention for given context X.

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context of events, same as input to fit and predict.

            y : array-like of type=int and shape=(n_samples,)
                Observed event.

            iterations : int, default=0
                Number of iterations to perform for optimization of actual
                event.

            batch_size : int, default=1024
                Batch size of items to optimize.

            verbose : boolean, default=False
                If True, prints progress.

            Returns
            -------
            prediction : torch.Tensor of shape=(n_samples,)
                Most likely predictions after explanation.

            confidence : torch.Tensor of shape=(n_samples,)
                Resulting confidence levels in y.

            attention : torch.Tensor of shape=(n_samples,)
                Optimal attention for predicting event y.
            """
        logger.info("explain {} samples".format(X.shape[0]))
        # Get unique samples
        X, y, inverse = unique_2d(X, y)
        logger.debug("explain reduced to {} samples".format(X.shape[0]))

        # Perform query
        confidence, attention, _ = self.context_builder.query(X, y,
            iterations = iterations,
            batch_size = batch_size,
            verbose    = verbose,
        )

        # Get prediction
        prediction = confidence.argmax(dim=1)
        # Compute confidence of y
        confidence = confidence[torch.arange(y.shape[0]), y.squeeze(1)]

        # Return prediction, confidence and attention
        return prediction[inverse], confidence[inverse], attention[inverse]

    ########################################################################
    #                          Score computation                           #
    ########################################################################

    def score(self, X, y, attention, beta=1):
        """Compute score based on event score and weighted context scores.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, len_context)
                Context scores.

            y : torch.Tensor of shape=(n_samples,)
                Event score.

            attention : torch.tensor of shape=(n_samples, len_context)
                Attention vector for score computation.

            beta : float, default=1
                Relative weight of context.

            Returns
            -------
            score : torch.Tensor of shape=(n_samples,)
                Maliciousness score of event including context.
            """
        logger.info("score {} samples".format(X.shape[0]))

        # Special case if beta == 0
        if beta == 0: return y.to(torch.float)

        # Cast to float
        X = X.to(torch.float)
        y = y.to(torch.float)

        for i, axis in enumerate(torch.unbind(X, dim=2)):
            X[:,:,i] = attention * axis

        # Return contextual maliciousness score
        return (y + beta*X.sum(dim=1)) / (1+beta)


    ########################################################################
    #                            Drawing method                            #
    ########################################################################

    def graph(self, clusters, fingerprints, scores, outfile):
        """Create a plot of fingerprints.

            Parameters
            ----------
            clusters : Array-like of shape=(n_samples,)
                Clusters to plot.

            fingerprints : Array-like of shape=(n_samples, n_features)
                Fingerprints to plot.

            scores : Array-like of shape=(n_samples,)
                Scores corresponding to clusters

            outfile : string
                Output file to write graph to.
            """
        from sklearn.metrics import pairwise_distances
        import networkx as nx
        from .utils import sp_unique
        import matplotlib.pyplot as plt
        from itertools import cycle

        # Initialise graph
        graph = nx.Graph()

        # Loop over each cluster
        for cluster, indices in lookup_table(clusters):
            # Continue if cluster is anomalous
            if cluster == -1 or indices.shape[0] < 5: continue

            # Get unique fingerprints
            fingerprints_, _, counts = sp_unique(fingerprints[indices])

            # Compute similarities
            similarity = np.round(
                1 - pairwise_distances(fingerprints_, metric='cityblock'),
                decimals = 4,
            )

            # Get scores
            scores_ = np.zeros(indices.shape[0], dtype=int)
            scores_[scores[indices, 0] >   0] = 1
            scores_[scores[indices, 0] >= 30] = 2
            scores_[scores[indices, 0] >= 70] = 3
            scores_[scores[indices, 1] >   0] = 4
            scores_ = dict(zip(*np.unique(scores_, return_counts=True)))
            scores_ = str(scores_) + ' MIXED' if len(scores_) > 1 else ''

            # Get initial size
            start = len(graph)

            # Add nodes to graph
            for i, (node, weight) in enumerate(zip(fingerprints_, counts)):
                graph.add_node(start+i, size=weight, label=scores_)
                # graph.nodes[start+i]["viz"] = {"size": int((weight**0.2) + 20)}

            # Add edges to graph
            for i in range(similarity.shape[0]):
                for j in range(i+1, similarity.shape[1]):
                    if similarity[i, j] >= 0.1:
                        graph.add_edge(start+i, start+j, weight=similarity[i, j])

        # Write file to output
        nx.write_gexf(graph, "{}.gexf".format(outfile))
        # nx.write_gml (graph, "{}.gml" .format(outfile))


        exit()

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_dict(self):
        """Return a pickle-compatible dictionary representation of the
            interpreter.

            Returns
            -------
            result : dict()
                JSON-compatible dictionary representation of the (trained)
                interpreter.
            """
        logger.info("to_dict")

        return {
            # Object parameters
            'features'       : self.features,
            'eps'            : self.eps,
            'min_samples'    : self.min_samples,
            'threshold'      : self.threshold,
            # Trained features
            'tree'           : self.tree,
            'labels'         : self.labels,
            'clusters'       : self.clusters,
            'scores'         : self.scores,
        }

    @classmethod
    def from_dict(cls, dictionary, context_builder=None):
        """Load the interpreter from the given dictionary.

            Parameters
            ----------
            dictionary : dict()
                Dictionary containing state information of the interpreter to
                load.

            context_builder : ContextBuilder, optional
                If given, use the given ContextBuilder for loading the
                Interpreter.

            Returns
            -------
            interpreter : Interpreter
                Interpreter, constructed from dictionary.
            """
        logger.info("from_dict")

        # Set context_builder if given separately
        if context_builder is not None:
            dictionary['context_builder'] = context_builder

        # List of required features
        features = {
            'context_builder': None,
            'features'       : 100,
            'eps'            : 0.1,
            'min_samples'    : 5,
            'threshold'      : 0.2,
            'tree'           : dict(),
            'labels'         : dict(),
            'clusters'       : list(),
            'scores'         : list(),
        }

        # Throw warning if dictionary does not contain values
        for feature, default in features.items():
            # Throw warning if feature not available
            if feature not in dictionary:
                warnings.warn("Loading interpreter from dictionary, required "
                              "feature '{}' not in dictionary. Defaulting to "
                              "default '{}'".format(feature, default))
                dictionary[feature] = default

        # Create new instance with given features
        result = cls(
            context_builder= dictionary.get('context_builder'),
            features       = dictionary.get('features') ,
            eps            = dictionary.get('eps'),
            min_samples    = dictionary.get('min_samples'),
            threshold      = dictionary.get('threshold'),
        )

        # Set tree and labels
        result.tree     = dictionary.get('tree')
        result.labels   = dictionary.get('labels')
        result.clusters = dictionary.get('clusters')
        result.scores   = dictionary.get('scores')

        # Return result
        return result

    def save(self, outfile):
        """Save model to output file.

            Parameters
            ----------
            outfile : string
                File to output model.
            """
        logger.info("save to {}".format(outfile))

        # Save to output file
        with open(outfile, 'wb') as outfile:
            pickle.dump(self.to_dict(), outfile)

    @classmethod
    def load(cls, infile, context_builder=None):
        """Load model from input file.

            Parameters
            ----------
            infile : string
                File from which to load model.

            context_builder : ContextBuilder, optional
                If given, use the given ContextBuilder for loading the
                Interpreter.

            Returns
            -------
            self : self
                Return self.
            """
        logger.info("load from {}".format(infile))

        # Load data
        with open(infile, 'rb') as infile:
            return Interpreter.from_dict(
                dictionary      = pickle.load(infile),
                context_builder = context_builder,
            )
