import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderAttention(nn.Module):

    def __init__(self, embedding, context_size, attention_size, num_layers=1,
                 dropout=0.1, bidirectional=False, LSTM=False):
        """Attention decoder for retrieving attention from context vector.

            Parameters
            ----------
            embedding : nn.Embedding
                Embedding layer to use.

            context_size : int
                Size of context to expect as input.

            attention_size : int
                Size of attention vector.

            num_layers : int, default=1
                Number of recurrent layers to use.

            dropout : float, default=0.1
                Default dropout rate to use.

            bidirectional : boolean, default=False
                If True, use bidirectional recurrent layer.

            LSTM : boolean, default=False
                If True, use LSTM instead of GRU.
            """
        # Call super
        super().__init__()

        ################################################################
        #                      Initialise layers                       #
        ################################################################
        # Embedding layer
        self.embedding = embedding

        # Recurrency layer
        self.recurrent = (nn.LSTM if LSTM else nn.GRU)(
            input_size    = embedding.embedding_dim,
            hidden_size   = context_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
        )

        # Attention layer
        self.attn = nn.Linear(
            in_features  = context_size * num_layers * (1+bidirectional),
            out_features = attention_size
            )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_vector, previous_input=None):
        """Compute attention based on input and hidden state.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, embedding_dim)
                Input from which to compute attention

            hidden : torch.Tensor of shape=(n_samples, hidden_size)
                Context vector from which to compute attention

            Returns
            -------
            attention : torch.Tensor of shape=(n_samples, context_size)
                Computed attention

            context_vector : torch.Tensor of shape=(n_samples, hidden_size)
                Updated context vector
            """
        # Get embedding from input
        embedded = self.embedding(previous_input)\
                   .view(-1, 1, self.embedding.embedding_dim)
        # Apply dropout layer
        embedded = self.dropout(embedded)

        # Compute attention and pass through hidden to next state
        attention, context_vector = self.recurrent(embedded, context_vector)
        # Apply dropout layer
        # attention = self.dropout(attention)
        # Compute attention
        attention = self.attn(attention.squeeze(1))
        # Normalise attention weights, i.e. sum to 1
        attention = F.softmax(attention, dim=1)

        # Return result
        return attention, context_vector



class DecoderEvent(nn.Module):

    def __init__(self, input_size, output_size, dropout=0.1):
        """"""
        # Call super
        super().__init__()

        # Initialise layers
        self.hidden  = nn.Linear(input_size, input_size)
        self.out     = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, attention):
        """Decode X with given attention.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, context_size, hidden_size)
                Input samples on which to apply attention.

            attention : torch.Tensor of shape=(n_samples, context_size)
                Attention to use for decoding step

            Returns
            -------
            output : torch.Tensor of shape=(n_samples, output_size)
                Decoded output
            """
        # Apply attention (by computing batch matrix-matrix product)
        attn_applied = torch.bmm(attention.unsqueeze(1), X).squeeze(1)
        # attn_applied = self.dropout(attn_applied)

        # Compute prediction based on latent dimension
        output = self.hidden(attn_applied).relu()
        # output = self.dropout(output)
        output = self.out(output)
        # Apply softmax for distribution
        output = F.log_softmax(output, dim=1)

        # Return result
        return output
