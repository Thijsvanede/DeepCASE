from   torch.autograd import Variable
import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):

    def __init__(self, size, smoothing=0.0):
        """Implements label smoothing loss function

            Parameters
            ----------
            size : int
                Number of labels

            smoothing : float, default=0.0
                Smoothing factor to apply
            """
        # Initialise super
        super(LabelSmoothing, self).__init__()
        # Set KL-Divergence loss function
        self.criterion      = nn.KLDivLoss(reduction='none')
        self.criterion_attn = nn.MSELoss()
        # Set size
        self.size = size
        # Set confidence and smoothing
        self.smoothing  =       smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target, weights=None, attention=None):
        """Forward data"""
        # Check if shape of data is the same as expected
        assert x.size(-1) == self.size
        # Set target if necessary
        if x.ndim != target.ndim:
            target = target.unsqueeze(-1)

        # Create true distribution
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target, self.confidence)
        # Apply criterion
        result = self.criterion(x, Variable(true_dist, requires_grad=False))
        # Apply weights if necessary
        if weights is not None:
            result *= weights.to(torch.float).unsqueeze(1)
        # Get result as sum
        result = result.sum()

        # Apply attention criterion if necessary
        if attention is not None:
            target = torch.ones(attention.shape, device=x.device) / attention.shape[1]
            return result + 0.05*self.criterion_attn(attention, target)
        else:
            return result
