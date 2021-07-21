import math
import numpy as np
import torch
from itertools       import cycle
from sklearn.metrics import confusion_matrix as cm


def confusion_report(
        y_true,
        y_pred,
        labels        = None,
        target_names  = None,
        sample_weight = None,
        normalize     = None,
        skip_x        = set(),
        skip_y        = set(),
    ):
    """Print the confusion matrix as a report.

        Parameters
        ----------
        y_true : array-like of shape=(n_samples,)
            Actual labels of evaluated values.

        y_pred : array-like of shape=(n_samples,)
            Predicted labels of evaluated values.

        labels : array-like of shape=(n_labels,), optional
            All different labels to include in the confusion report.
            If none are given, labels are inferred from the y_true and y_pred
            inputs. Labels should at least contain all labels in y_true and
            y_pred, but additional labels may be given.

        target_names : array-like of shape=(n_labels,), optional
            If given, use names pressented by target_names for each
            corresponding label.

        sample_weight : array-like of shape=(n_samples,), optional
            If given, weigh input by given sample_weight.

        normalize : {‘true’, ‘pred’, ‘all’}, default=None
            Normalizes confusion matrix over the true (rows), predicted
            (columns) conditions or all the population. If None, confusion
            matrix will not be normalized.

        skip_x : set(), optional
            Set of target_names to skip while printing the columns.

        skip_y : set(), optional
            Set of target_names to skip while printing the rows.

        Returns
        -------
        result : string
            Report detailing the confusion matrix for a given prediction.
        """
    # Compute matrix
    matrix = cm(
        y_true        = y_true,
        y_pred        = y_pred,
        labels        = labels,
        sample_weight = sample_weight,
        normalize     = normalize,
    )

    if target_names is not None:
        assert labels       is not None
        assert target_names is not None
        assert len(labels) == len(target_names)

        # Add labels to matrix
        matrix = np.concatenate(([target_names], matrix))
        matrix = np.concatenate(([["T\\P"] + target_names], matrix.T)).T

    # Compute width of rows
    width = np.vectorize(len)(matrix).max()

    # Transform to string
    result = ""
    mask_x = [i for i, x in enumerate(matrix[0   ]) if x not in skip_x]
    mask_y = [i for i, x in enumerate(matrix[:, 0]) if x not in skip_y]
    for row in matrix[mask_y]:
        result += "\t".join(
            "{:>{width}}".format(element, width=width)
            for element in row[mask_x]
        ) + '\n'

    return result



def show_sequences(context, events, labels=None, mapping=None, NO_EVENT=None, maximum=40):
    """Print all generated sequences.

        Parameters
        ----------
        context : torch.Tensor of shape=(n_samples, len_context)
            Context tensor

        events : torch.Tensor of shape=(n_samples,)
            Event tensor

        labels : torch.Tensor of shape=(n_samples,), optional
            Labels corresponding to sequences

        mapping : dict()
            Mapping to rewrite actual names of events

        NO_EVENT : object, optional
            The NO_EVENT object to ignore displaying

        maximum : int, default=40
            Maximum number of samples to show
    """
    ########################################################################
    #                             Preparation                              #
    ########################################################################

    # Get dimensions
    n_samples, n_features = context.shape

    # Only print start and end (if necessary)
    shortened = False
    if n_samples > maximum:
        context_a = context[: math.ceil (maximum / 2) ]
        context_b = context[ -math.floor(maximum / 2):]
        context   = torch.cat((context_a, context_b), axis=0)

        events_a = events[: math.ceil (maximum / 2) ]
        events_b = events[ -math.floor(maximum / 2):]
        events   = torch.cat((events_a, events_b), axis=0)

        if labels is not None:
            labels_a = labels[: math.ceil (maximum / 2) ]
            labels_b = labels[ -math.floor(maximum / 2):]
            labels   = torch.cat((labels_a, labels_b), axis=0)

        # Set shortened to True
        shortened = True


    # Vectorize functions
    vmap    = np.vectorize(lambda x: str(mapping.get(x, '?'))) # Mapping
    vlength = np.vectorize(lambda x: len(str(x)))              # String length

    # Add labels if None
    if labels is None:
        length_l = max(len(str(labels)), 5)
        labels = cycle([None])
    else:
        length_l = max(vlength(labels).max(), 5)

    # Apply mapping if necessary
    if mapping is not None:
        # If NO_EVENT is given add it to mapping
        for key, value in mapping.items():
            if value == NO_EVENT:
                mapping[key] = ""

        # Apply mapping to context and events
        context = vmap(context.cpu().numpy())
        events  = vmap(events .cpu().numpy())

    # Compute context lengths
    lengths_c = vlength(context)
    lengths_c = np.array([
        lengths_c[:, column].max()
        for column in range(context.shape[1])
    ])

    # Compute event lengths
    length_e = max(vlength(events).max(), 7)

    ########################################################################
    #                           Print formatting                           #
    ########################################################################

    # Print header
    print("\n|{}{}{}| [{:^}]".format(
        "-" * math.ceil ((lengths_c.sum() + 2*(lengths_c.shape[0]-1) + length_e - 1) / 2),
        "Sequence",
        "-" * math.floor((lengths_c.sum() + 2*(lengths_c.shape[0]-1) + length_e - 1) / 2),
        "Label",
    ))
    print("|{}{}{}|     |{}{}{}|".format(
        "-" * math.ceil ((lengths_c.sum() + 2*(lengths_c.shape[0]-1) - 7) / 2),
        "Context",
        "-" * math.floor((lengths_c.sum() + 2*(lengths_c.shape[0]-1) - 7) / 2),

        "-" * math.ceil ((length_e - 5) / 2),
        "Event",
        "-" * math.floor((length_e - 5) / 2),
    ))

    # Print all sequences
    for index, (ctx, event, label) in enumerate(zip(context, events, labels)):
        if shortened and index == math.ceil (maximum / 2):
            print(' {:^{width_context}}       {:^{width_event}} '.format(
                '⋮',
                '⋮',
                width_context = lengths_c.sum() + 2*(lengths_c.shape[0]-1),
                width_event   = length_e,
            ))

        print("[{}] -->  {:^{width_event}}  [{:^{width_label}}]".format(
            ", ".join("{:{width}}".format(c, width=l) for c, l in zip(ctx, lengths_c)),
            event,
            str(label),
            width_event = length_e,
            width_label = length_l,
        ))

    # Print footer
    print("|{}{}{}|     |{}{}{}|".format(
        "-" * math.ceil ((lengths_c.sum() + 2*(lengths_c.shape[0]-1) - 7) / 2),
        "Context",
        "-" * math.floor((lengths_c.sum() + 2*(lengths_c.shape[0]-1) - 7) / 2),

        "-" * math.ceil ((length_e - 5) / 2),
        "Event",
        "-" * math.floor((length_e - 5) / 2),
    ))
    print("|{}{}{}| [{:^}]\n".format(
        "-" * math.ceil ((lengths_c.sum() + 2*(lengths_c.shape[0]-1) + length_e - 1) / 2),
        "Sequence",
        "-" * math.floor((lengths_c.sum() + 2*(lengths_c.shape[0]-1) + length_e - 1) / 2),
        "Label",
    ))
    print("Found {} sequences of context length {}".format(
        n_samples,
        n_features,
    ))
