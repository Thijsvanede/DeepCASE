import numpy as np

def probability_non_benign(malicious, benign, observations=1):
    """Compute the probability of seeing any malicious sample given a cluster.

        Parameters
        ----------
        malicious : array-like of shape=(n_clusters,)
            Number of malicious samples per cluster.

        benign : array-like of shape=(n_clusters,)
            Number of benign samples per cluster.

        observations : int, default=1
            Number of observations for which to compute probability.

        Returns
        -------
        probability : float
            Probability of finding non-benign cluster.
        """
    # Get arrays as numpy
    malicious = np.asarray(malicious)
    benign    = np.asarray(benign)
    # Get total # samples
    total     = benign + malicious

    # Get benign probability
    probability = probability_benign(
        malicious    = malicious,
        benign       = benign,
        observations = observations,
    )

    return (1 - probability).sum() / total.shape[0]

def probability_mixed(malicious, benign, observations=1):
    """Compute the probability of seeing mixed samples given a cluster.

        Parameters
        ----------
        malicious : array-like of shape=(n_clusters,)
            Number of malicious samples per cluster.

        benign : array-like of shape=(n_clusters,)
            Number of benign samples per cluster.

        observations : int, default=1
            Number of observations for which to compute probability.

        Returns
        -------
        probability : float
            Probability of finding mixed cluster.
        """
    # Get arrays as numpy
    malicious = np.asarray(malicious)
    benign    = np.asarray(benign)
    # Get total # samples
    total     = benign + malicious

    # Get benign probability
    p_benign = probability_benign(
        malicious    = malicious,
        benign       = benign,
        observations = observations,
    )

    # Get benign probability
    p_malicious = probability_malicious(
        malicious    = malicious,
        benign       = benign,
        observations = observations,
    )

    return (1 - p_benign - p_malicious).sum() / total.shape[0]


def probability_benign(malicious, benign, observations=1):
    """Compute the probability of seeing only benign samples given a cluster.

        Parameters
        ----------
        malicious : array-like of shape=(n_clusters,)
            Number of malicious samples per cluster.

        benign : array-like of shape=(n_clusters,)
            Number of benign samples per cluster.

        observations : int, default=1
            Number of observations for which to compute probability.

        Returns
        -------
        probability : float
            Probability of finding only benign samples in mixed cluster.
        """
    # Get arrays as numpy
    malicious = np.asarray(malicious)
    benign    = np.asarray(benign)
    # Get total # samples
    total     = benign + malicious

    # Set probability of finding a benign cluster
    probability = np.ones(total.shape[0])

    # For each observation, get the probability of finding only benign samples
    for observation in range(observations):
        benign_ = benign - observation
        total_  = total  - observation
        mask = benign_ > 0
        probability[ mask] *= benign_[mask] / total_[mask]
        probability[~mask]  = 0

    # Return probability
    return probability


def probability_malicious(malicious, benign, observations=1):
    """Compute the probability of seeing only malicious samples given a cluster.

        Parameters
        ----------
        malicious : array-like of shape=(n_clusters,)
            Number of malicious samples per cluster.

        benign : array-like of shape=(n_clusters,)
            Number of benign samples per cluster.

        observations : int, default=1
            Number of observations for which to compute probability.

        Returns
        -------
        probability : float
            Probability of finding only malicious samples in mixed cluster.
        """
    # Get arrays as numpy
    malicious = np.asarray(malicious)
    benign    = np.asarray(benign)
    # Get total # samples
    total     = benign + malicious

    # Set probability of finding a benign cluster
    probability = np.ones(total.shape[0])

    # For each observation, get the probability of finding only benign samples
    for observation in range(observations):
        malicious_ = malicious - observation
        total_     = total     - observation
        mask = malicious_ > 0
        probability[ mask] *= malicious_[mask] / total_[mask]
        probability[~mask]  = 0

    # Return probability
    return probability



if __name__ == "__main__":
    # Mixed clusters malicious samples
    breaches_malicious = np.asarray([
        5, 10, 77, 43, 46, 7, 17, 6, 11, 5, 10, 2929, 4, 26, 8, 22, 652, 8, 56, 1, 10, 13, 35
    ])
    # Mixed clusters benign samples
    breaches_benign    = np.asarray([
        4, 10, 2, 4, 7, 2, 3, 1, 1, 2, 1, 295, 1, 1, 2, 2, 71, 2, 10, 5, 2, 8, 1
    ])

    total_observations = 100

    print("Non-benign")
    for observations in range(1, total_observations+1):
        print("({:2}, {:.4f})".format(
            observations, probability_non_benign(
            malicious    = breaches_malicious,
            benign       = breaches_benign,
            observations = observations,
        )))

    print("\nMixed")
    for observations in range(1, total_observations+1):
        print("({:2}, {:.4f})".format(
            observations, probability_mixed(
            malicious    = breaches_malicious,
            benign       = breaches_benign,
            observations = observations,
        )))
