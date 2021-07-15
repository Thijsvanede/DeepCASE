from sklearn.cluster import dbscan
import logging
import scipy.sparse as sp

from .utils import sp_unique

# Set logger
logger = logging.getLogger(__name__)

class Cluster(object):

    def __init__(self, p=2):
        """Initialise clustering for minkowski distence using given value of p.

            Parameters
            ----------
            p : float, default=2
                The power of the Minkowski metric to be used to calculate
                distance between points.
            """
        # Initialise parameters
        self.p = p

    def dbscan(self, X, eps=0.1, min_samples=5, algorithm='auto', verbose=False):
        """Perform optimized version of DBSCAN.

            Parameters
            ----------
            X : sparse array-like of shape=(n_samples, n_features)
                Samples to cluster using DBSCAN

            eps : float, default=0.1
                Epsilon to use for DBSCAN algorithm

            min_samples : int, default=5
                Minimum number of samples to use for DBSCAN clustering

            algorithm : 'auto'|'similar'|'unique', default='auto'
                DBSCAN optimisation to use. If 'auto', automatically choose
                between 'similar' and 'unique' depending on several heuristics.

                Note
                ----
                Algorithms give equivalent (i.e., ARI=1) results, but differ in
                speed. For large datasets, 'similar' gives better results.

            Returns
            -------
            clusters : np.array of shape=(n_samples,)
                Clusters from DBSCAN
            """
        # Ensure array is sparse
        assert sp.issparse(X), "X should be a sparse array"

        # Get unique rows
        X, inverse, weights = sp_unique(X)

        # Perform DBSCAN
        _, result = dbscan(X,
            eps           = eps,
            min_samples   = min_samples,
            sample_weight = weights,
            p             = self.p,
            n_jobs        = 1 if X.shape[0] < 5e3 else -3,
        )

        # Return result
        return result[inverse]
