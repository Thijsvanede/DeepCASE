import torch

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
