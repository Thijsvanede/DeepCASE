import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

################################################################################
#                            Sparse unique function                            #
################################################################################

def sp_unique(sp_matrix, axis=0):
    """Returns a sparse matrix with the unique rows (axis=0)
        or columns (axis=1) of an input sparse matrix sp_matrix.

        Parameters
        ----------
        sp_matrix : scipy.sparse of shape=(n_samples, n_features)
            Sparse for which to compute unique rows.

        axis : int, defaul=0
            Axis along which to compute uniqueness.

        Returns
        -------
        unique : sp_matrix scipy.sparse
            Unique samples.

        inverse : np.array of shape=(n_samples,)
            Inverse to reconstruct original sp_matrix.
        """
    if axis == 1:
        sp_matrix = sp_matrix.T

    old_format = sp_matrix.getformat()
    dt         = np.dtype(sp_matrix)
    ncols      = sp_matrix.shape[1]

    if old_format != 'lil':
        sp_matrix = sp_matrix.tolil()

    _, ind, inverse, count = np.unique(sp_matrix.data + sp_matrix.rows,
        return_index   = True,
        return_inverse = True,
        return_counts  = True,
    )
    rows = sp_matrix.rows[ind]
    data = sp_matrix.data[ind]
    nrows_uniq = data.shape[0]

    sp_matrix = sp.lil_matrix((nrows_uniq, ncols), dtype=dt)  #  or sp_matrix.resize(nrows_uniq, ncols)
    sp_matrix.data = data
    sp_matrix.rows = rows

    ret = sp_matrix.asformat(old_format)
    if axis == 1:
        ret = ret.T
    return ret, inverse, count

################################################################################
#                                 Lookup Table                                 #
################################################################################

def lookup_table(X, hash=lambda x: x.item(), key=lambda x: x, verbose=False):
    """Create a lookup table for unique items in X

        Parameters
        ----------
        X : array-like of shape=(n_samples,)
            Array for which to create lookup table.

        key : func, default=lambda x: x.item()
            Key function to apply to each label

        verbose : boolean, default=False
            If True, print progress

        Returns
        -------
        result : dict()
            Dictionary of unique_value -> list of indices
        """
    # Initialise lookup table
    lookup_table = dict()

    # Add progress bar if required
    if verbose: X = tqdm(X, desc="Lookup table")

    # Loop over items in table
    for index, label in enumerate(X):
        hashed = hash(label)
        # Add label to lookup table if it does not exist
        if hashed not in lookup_table:
            lookup_table[hashed] = [key(label), list()]
        # Append item
        lookup_table[hashed][1].append(index)

    # Return result as iterator over numpy array
    return [(v1, np.asarray(v2)) for v1, v2 in lookup_table.values()]

################################################################################
#                                  Unique 2D                                   #
################################################################################

def unique_2d(*X):
    """Get the unique combinations of inputs X.

        Parameters
        ----------
        X : array-like of type=int and shape=(n_samples, n_features)
            Input events for which to get unique combinations

        Returns
        -------
        *X_unique : np.array of shape=(n_samples_unique, n_features)
            Unique input event combinations

        inverse : np.array of shape=(n_samples,)
            Inverse used to reconstruct original values
        """
    # Get input shapes
    shapes = [x.shape[1] for x in X]

    # Get unique combined inputs
    unique, inverse = torch.unique(
        torch.cat(X, dim=1),
        dim            = 0,
        return_inverse = True
    )

    # Retrieve original inputs
    result = list()
    previous = 0
    for shape in shapes:
        result.append(unique[:, previous:previous+shape])
        previous += shape

    # Add inverse
    result.append(inverse)

    # Return result
    return tuple(result)
