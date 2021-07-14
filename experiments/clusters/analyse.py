import argformat
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import homogeneity_score
from deepcase.context_builder   import ContextBuilder
from deepcase.interpreter       import Interpreter
from deepcase.interpreter.utils import lookup_table


def n_choose_k(n, k):
    if k > n: return 0
    return math.factorial(n) / (math.factorial(n-k) * math.factorial(k))

def probability_suspicious(classes, k):
    # Compute total
    N = classes.sum(axis=1)

    top = np.zeros(N.shape[0])
    # Compute factorials
    for c in range(classes.shape[1]):
        top += scipy.special.comb(classes[:, c], k)

    # Set result
    result = top / scipy.special.comb(N, k)
    # Remove nan values
    result[np.isnan(result)] = 0
    # Return result
    return 1 - result

def probability_high_risk(classes, k):
    # Compute total
    N = classes.sum(axis=1)

    # Compute non highest class
    processed = np.zeros(classes.shape[0], dtype=bool)
    top       = np.zeros(classes.shape[0], dtype=float)
    for c in reversed(range(classes.shape[1])):
        # Get mask of relevant items
        mask = classes[:, c] > 0

        # Get relevant mask
        mask_ = np.logical_and(~processed, mask)
        # Compute top
        top[mask_] = classes[mask_, :c].sum(axis=1)

        # Set processed items
        processed[mask] = True

    result = scipy.special.comb(top, k) / scipy.special.comb(N, k)
    # Remove nan values
    result[np.isnan(result)] = 0

    # Return result
    return 1 - result

def probability_either(classes, k):
    # Compute total
    N = classes.sum(axis=1)

    top = np.zeros(N.shape[0])
    highest = np.zeros(classes.shape[0], dtype=bool)
    # Compute factorials
    for c in reversed(range(classes.shape[1])):
        # If label is highest
        top[highest] += scipy.special.comb(classes[highest, c], k)

        # Add highest as allowed
        highest[classes[:, c] > 0] = True

    # Set result
    result = top / scipy.special.comb(N, k)
    # Remove nan values
    result[np.isnan(result)] = 0
    # Return result
    return 1 - result



def analyse(interpreter):
    """Perform cluster analysis from paper."""
    # Extract clusters and scores
    clusters  = interpreter.clusters[interpreter.clusters != -1]
    scores    = np.zeros(interpreter.scores.shape[0])

    # Set risk levels from scores
    scores[:] = interpreter.scores[:, 0]
    scores[interpreter.scores[:, 0]  >= 0] = 0
    scores[interpreter.scores[:, 0]  >  0] = 1
    scores[interpreter.scores[:, 0]  >  1] = 2
    scores[interpreter.scores[:, 0]  >  2] = 3
    scores[interpreter.scores[:, 0]  >  3] = 4
    scores = scores.astype(int)

    matrix = list()

    # Initialise indices of risk levels
    indices_levels = [None] * 6
    clusters_adjacent = list()
    indices_adjacent  = list()
    # Loop over all clusters
    for cluster, indices in lookup_table(clusters):
        # Check scores
        scores_, counts_ = np.unique(scores[indices], return_counts=True)
        # If scores are unique, add indices to specific level
        if scores_.shape[0] == 1:
            if indices_levels[scores_[0]] is None:
                indices_levels[scores_[0]] = list()
            indices_levels[scores_[0]].extend(indices)
        # Otherwise add to SUSPICIOUS indices
        else:
            # Case of adjacent indices
            if scores_.shape[0] == 2 and scores_[1] - scores_[0] == 1:
                clusters_adjacent.append(cluster)
                indices_adjacent.extend(indices)
            if indices_levels[-1] is None:
                indices_levels[-1] = list()
            indices_levels[-1].extend(indices)

        # Add to matrix
        entry = [0, 0, 0, 0, 0]
        for score_, count_ in zip(scores_, counts_):
            entry[score_] += count_

        matrix.append(entry)

    # Print header
    print("{:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Risk level", "Clusters", "Total", "Average", "Min", "Max", "SD"
    ))

    # Compute results
    for risk_level, indices in zip(['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK', 'SUSPICIOUS'], indices_levels):
        # Extract clusters for risk level
        clusters_risk = clusters[indices]

        # Get unique and counts
        unique, counts = np.unique(clusters_risk, return_counts=True)
        # Compute and print statistics per risk level
        print("{:10} {:8} {:8} {:8.1f} {:8} {:8} {:8.1f}".format(
            risk_level, unique.shape[0], counts.sum(), counts.mean(), counts.min(),
            counts.max(), counts.std(),
        ))

    # Compute statistics for total
    unique, counts = np.unique(clusters, return_counts=True)

    # Print total
    print("{:10} {:8} {:8} {:8.1f} {:8} {:8} {:8.1f}".format(
        'Total', unique.shape[0], counts.sum(), counts.mean(), counts.min(),
        counts.max(), counts.std(),
    ))

    # Print covered percentages
    print()
    print("Single level clusters   : {:8}/{:8} = {:8.4f}%".format(
        unique.shape[0] - np.unique(clusters[indices_levels[-1]]).shape[0],
        unique.shape[0],
        100 * (unique.shape[0] - np.unique(clusters[indices_levels[-1]]).shape[0]) / unique.shape[0],
    ))
    print("Single level sequences  : {:8}/{:8} = {:8.4f}%".format(
        clusters.shape[0] - len(indices_levels[-1]),
        clusters.shape[0],
        100 * (clusters.shape[0] - len(indices_levels[-1])) / clusters.shape[0],
    ))
    print("Adjacent level clusters : {:8}/{:8} = {:8.4f}%".format(
        unique.shape[0] - np.unique(clusters[indices_levels[-1]]).shape[0] + len(clusters_adjacent),
        unique.shape[0],
        100 * (unique.shape[0] - np.unique(clusters[indices_levels[-1]]).shape[0] + len(clusters_adjacent)) / unique.shape[0],
    ))
    print("Adjacent level sequences: {:8}/{:8} = {:8.4f}%".format(
        clusters.shape[0] - len(indices_levels[-1]) + len(indices_adjacent),
        clusters.shape[0],
        100 * (clusters.shape[0] - len(indices_levels[-1]) + len(indices_adjacent)) / clusters.shape[0],
    ))

    # Compute homogeneity
    print("\nHomogeneity: {:.4f}\n".format(homogeneity_score(
        scores, clusters
    )))

    # Compute probability of identifying clusters

    # Get matrix as array
    matrix = np.asarray(matrix)
    matrix = matrix[(matrix > 0).sum(axis=1) > 1] # Only use SUSPICIOUS clusters

    print("SUSPICIOUS")
    for i in range(1, 21):
        avg_prob = probability_suspicious(matrix, i)
        print("({:2}, {:.4f})".format(i, np.average(avg_prob)))
    print()


    print("CONSERVATIVE")
    for i in range(1, 21):
        avg_prob = probability_high_risk(matrix, i)
        print("({:2}, {:.4f})".format(i, np.average(avg_prob)))
    print()

    print("EITHER")
    for i in range(1, 300):
        avg_prob = probability_either(matrix, i)
        weights = matrix.sum(axis=1)
        print("({:2}, {:.4f})".format(i, np.average(avg_prob)))

    print('\n\n')

    # Print distribution of cluster sizes
    _, sizes = np.unique(
        clusters,
        return_counts = True,
    )

    for threshold in range(5, 51):
        print("({}, {})".format(threshold, np.sum(sizes == threshold)))
    print("(51, -1)")
    print("(57, {})".format(np.sum(np.logical_and(sizes > 50, sizes <= 100))))
    print("(59, -1)")
    print("(62, {})".format(np.sum(np.logical_and(sizes > 100, sizes <= 1_000))))
    print("(64, -1)")
    print("(67, {})".format(np.sum(np.logical_and(sizes > 1_000, sizes <= 10_000))))
    print("(69, -1)")
    print("(72, {})".format(np.sum(sizes > 10_000)))
    print("(74, -1)")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = "Analyse clusters from Interpreter",
        formatter_class = argformat.StructuredFormatter,
    )

    parser.add_argument('builder'    , help='file from which to load ContextBuilder')
    parser.add_argument('interpreter', help='file from which to load Interpreter')
    args = parser.parse_args()

    # Load objects
    context_builder = ContextBuilder.load(args.builder, 'cpu')
    interpreter     = Interpreter.load(args.interpreter, context_builder)

    # Perform analysis
    analyse(interpreter)
