import math
import numpy as np
from itertools import cycle

def show_sequences(context, events, labels=None, mapping=None, NO_EVENT=None):
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
    """

    ########################################################################
    #                             Preparation                              #
    ########################################################################

    # Vectorize functions
    vmap = np.vectorize(lambda x: str(mapping.get(x, '?'))) # Mapping
    vlength = np.vectorize(lambda x: len(x))                # String length

    # Add labels if None
    if labels is None:
        length_l = max(len(str(labels)), 5)
        labels = cycle([None])
    else:
        length_l = max(vlength(labels), 5)

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
    for ctx, event, label in zip(context, events, labels):
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
    print("Found {} sequences of context length {}".format(context.shape[0], context.shape[1]))
