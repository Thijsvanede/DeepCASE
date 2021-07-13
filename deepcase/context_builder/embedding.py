import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingOneHot(nn.Module):
    """Embedder using simple one hot encoding."""

    def __init__(self, input_size):
        """Embedder using simple one hot encoding.

            Parameters
            ----------
            input_size : int
                Maximum number of inputs to one_hot encode
            """
        super().__init__()
        self.input_size    = input_size
        self.embedding_dim = input_size

    def forward(self, X):
        """Create one-hot encoding of input

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples,)
                Input to encode.

            Returns
            -------
            result : torch.Tensor of shape=(n_samples, input_size)
                One-hot encoded version of input
            """
        return F.one_hot(X, self.input_size).to(torch.float)
