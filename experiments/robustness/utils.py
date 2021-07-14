import numpy as np
from sklearn.metrics import classification_report

def cm_to_cp(matrix, digits=4, target_names=None, zero_division=0):
    """Transform confusion matrix into classification report.

        Parameters
        ----------
        matrix : array-like of shape=(actual_classes, predicted_classes)
            Confusion matrix to transform into classification report.

        Returns
        -------
        result : string
            Classification report.
        """
    # Transform to numpy array
    matrix = np.asarray(matrix)

    # Get labels
    classes_true, classes_pred = matrix.shape
    true, pred = np.array(np.meshgrid(
        np.arange(classes_true),
        np.arange(classes_pred),
    )).T.reshape(-1, 2).T

    return classification_report(true, pred,
        digits        = digits,
        target_names  = target_names,
        sample_weight = matrix.flatten(),
        zero_division = zero_division,
    )
