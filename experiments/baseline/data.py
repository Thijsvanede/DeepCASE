import torch

def prepare_y(incidents, breaches):
    # Initialise result
    result = torch.zeros(
        incidents.shape[0],
        requires_grad = False,
        dtype         = torch.long,
    )

    # Flatten
    incidents = incidents.squeeze(1)
    breaches  = breaches .squeeze(1)

    # Set results
    result[incidents >   0] = 1
    result[incidents >= 30] = 2
    result[incidents >= 70] = 3
    result[breaches  >   0] = 4

    # Return result
    return result
