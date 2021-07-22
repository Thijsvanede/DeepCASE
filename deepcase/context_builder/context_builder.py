# Imports
import logging
import math
import random
from tqdm import tqdm

# Torch imports
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
from torch.autograd   import Variable
from torch.utils.data import DataLoader, TensorDataset

# Custom package imports
from .decoders  import DecoderAttention, DecoderEvent
from .embedding import EmbeddingOneHot
from .encoders  import Encoder
from .loss      import LabelSmoothing
from .utils     import unique_2d

# Set logger
logger = logging.getLogger(__name__)

class ContextBuilder(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=128, num_layers=1,
                 max_length=10, bidirectional=False, LSTM=False):
        """ContextBuilder that learns to interpret context from security events.
            Based on an attention-based Encoder-Decoder architecture.

            Parameters
            ----------
            input_size : int
                Size of input vocabulary, i.e. possible distinct input items

            output_size : int
                Size of output vocabulary, i.e. possible distinct output items

            hidden_size : int, default=128
                Size of hidden layer in sequence to sequence prediction.
                This parameter determines the complexity of the model and its
                prediction power. However, high values will result in slower
                training and prediction times

            num_layers : int, default=1
                Number of recurrent layers to use

            max_length : int, default=10
                Maximum lenght of input sequence to expect

            bidirectional : boolean, default=False
                If True, use a bidirectional encoder and decoder

            LSTM : boolean, default=False
                If True, use an LSTM as a recurrent unit instead of GRU
            """
        logger.info("ContextBuilder.__init__")

        # Initialise super
        super().__init__()

        ################################################################
        #                      Initialise layers                       #
        ################################################################

        # Create embedding
        self.embedding         = nn.Embedding(input_size, hidden_size)
        self.embedding_one_hot = EmbeddingOneHot(input_size)

        # Create encoder
        self.encoder = Encoder(
            embedding     = self.embedding_one_hot,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            bidirectional = bidirectional,
            LSTM          = LSTM
        )

        # Create attention decoder
        self.decoder_attention = DecoderAttention(
            embedding      = self.embedding,
            context_size   = hidden_size,
            attention_size = max_length,
            num_layers     = num_layers,
            dropout        = 0.1,
            bidirectional  = bidirectional,
            LSTM           = LSTM,
        )

        # Create event decoder
        self.decoder_event = DecoderEvent(
            input_size  = input_size,
            output_size = output_size,
            dropout     = 0.1,
        )

    ########################################################################
    #                        ContextBuilder Forward                        #
    ########################################################################

    def forward(self, X, y=None, steps=1, teach_ratio=0.5):
        """Forwards data through ContextBuilder.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_len)
                Tensor of input events to forward.

            y : torch.Tensor of shape=(n_samples, steps), optional
                If given, use value of y as next input with probability
                teach_ratio.

            steps : int, default=1
                Number of steps to predict in the future.

            teach_ratio : float, default=0.5
                Ratio of sequences to train that use given labels Y.
                The remaining part will be trained using the predicted values.

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, steps, output_size)
                The confidence level of each output event.

            attention : torch.Tensor of shape=(n_samples, steps, seq_len)
                Attention corrsponding to X given as (batch, out_seq, in_seq).
            """
        logger.info("forward {} samples".format(X.shape[0]))

        ####################################################################
        #                   Perform check on events in X                   #
        ####################################################################

        if X.max() >= self.embedding_one_hot.input_size:
            raise ValueError(
                "Expected {} different input events, but received input event "
                "'{}' not in expected range 0-{}. Please ensure that the "
                "ContextBuilder is configured with the correct input_size and "
                "output_size".format(
                self.embedding_one_hot.input_size,
                X.max(),
                self.embedding_one_hot.input_size-1,
            ))

        ####################################################################
        #                           Forward data                           #
        ####################################################################


        # Initialise results
        confidence = list()
        attention  = list()

        # Get initial inputs of decoder
        decoder_input  = torch.zeros(
            size       = (X.shape[0], 1),
            dtype      = torch.long,
            device     = X.device,
        )

        # Encode input
        X_encoded, context_vector = self.encoder(X)

        # Loop over all targets
        for step in range(steps):
            # Compute attention
            attention_, context_vector = self.decoder_attention(
                context_vector = context_vector,
                previous_input = decoder_input,
            )

            # Compute event probability distribution
            confidence_ = self.decoder_event(
                X         = X_encoded,
                attention = attention_,
            )

            # Store confidence
            confidence.append(confidence_)
            # Store attention
            attention .append(attention_)

            # Detatch from history
            if y is not None and random.random() <= teach_ratio:
                decoder_input = y[:, step]
            else:
                decoder_input = confidence_.argmax(dim=1).detach().unsqueeze(1)

        # Return result
        return torch.stack(confidence, dim=1), torch.stack(attention, dim=1)


    ########################################################################
    #                         Fit/predict methods                          #
    ########################################################################

    def fit(self, X, y, epochs=10, batch_size=128, learning_rate=0.01,
            optimizer=optim.SGD, teach_ratio=0.5, verbose=True):
        """Fit the sequence predictor with labelled data

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context to train with.

            y : array-like of type=int and shape=(n_samples, n_future_events)
                Sequences of target events.

            epochs : int, default=10
                Number of epochs to train with.

            batch_size : int, default=128
                Batch size to use for training.

            learning_rate : float, default=0.01
                Learning rate to use for training.

            optimizer : optim.Optimizer, default=torch.optim.SGD
                Optimizer to use for training.

            teach_ratio : float, default=0.5
                Ratio of sequences to train including labels.

            verbose : boolean, default=True
                If True, prints progress.

            Returns
            -------
            self : self
                Returns self
            """
        logger.info("fit {} samples".format(X.shape[0]))

        # Get current mode
        mode = self.training
        # Get input as torch tensors
        device = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.int64, device=device)
        y = torch.as_tensor(y, dtype=torch.int64, device=device)

        # Set to training mode
        self.train()

        # Set criterion and optimiser
        criterion = LabelSmoothing(self.decoder_event.out.out_features, 0.1)
        optimizer = optimizer(
            params = self.parameters(),
            lr     = learning_rate
        )

        # Load dataset
        data = DataLoader(TensorDataset(X, y),
            batch_size = batch_size,
            shuffle    = True,
        )

        # Loop over each epoch
        for epoch in range(1, epochs+1):
            try:
                # Set progress bar if necessary
                if verbose:
                    data = tqdm(data,
                        desc="[Epoch {:{width}}/{:{width}} loss={:.4f}]"
                        .format(epoch, epochs, 0, width=len(str(epochs)))
                    )

                # Set average loss
                total_loss  = 0
                total_items = 0

                # Loop over entire dataset
                for X_, y_ in data:
                    # Clear gradients
                    optimizer.zero_grad()

                    # Get prediction
                    confidence, _ = self.forward(X_, y_,
                        steps       = y_.shape[1],
                        teach_ratio = teach_ratio
                    )

                    # Compute loss
                    loss = 0
                    for step in range(confidence.shape[1]):
                        loss += criterion(confidence[:, step], y_[:, step])

                    # Backpropagate
                    loss.backward()
                    optimizer.step()

                    # Update description
                    total_loss  += loss.item() / X_.shape[1]
                    total_items += X_.shape[0]

                    if verbose:
                        data.set_description(
                            "[Epoch {:{width}}/{:{width}} loss={:.4f}]"
                            .format(epoch, epochs, total_loss/total_items,
                            width=len(str(epochs))))

            except KeyboardInterrupt as e:
                print("\nTraining interrupted, performing clean stop")
                break

        # Reset to original mode
        self.train(mode)

        # Return self
        return self


    def predict(self, X, y=None, steps=1):
        """Predict the next elements in sequence.

            Parameters
            ----------
            X : torch.Tensor
                Tensor of input sequences

            y : ignored

            steps : int, default=1
                Number of steps to predict into the future

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, seq_len, output_size)
                The confidence level of each output

            attention : torch.Tensor of shape=(n_samples, input_length)
                Attention corrsponding to X given as (batch, out_seq, seq_len)
            """
        logger.info("predict {} samples".format(X.shape[0]))

        # Get current mode
        mode = self.training
        # Set to prediction mode
        self.eval()

        # Memory optimization, only use unique values
        X, inverse = torch.unique(X, dim=0, return_inverse=True)

        logger.info("predict {}/{} unique samples".format(X.shape[0], inverse.shape[0]))

        # Do not perform gradient descent
        with torch.no_grad():
            # Perform all in single batch
            confidence, attention = self.forward(X, steps=steps)

        # Reset to original mode
        self.train(mode)

        # Return result
        return confidence[inverse], attention[inverse]


    def fit_predict(self, X, y, epochs=10, batch_size=128, learning_rate=0.01,
                    optimizer=optim.SGD, teach_ratio=0.5, verbose=True):
        """Fit the sequence predictor with labelled data

            Parameters
            ----------
            X : torch.Tensor
                Tensor of input sequences

            y : torch.Tensor
                Tensor of output sequences

            epochs : int, default=10
                Number of epochs to train with

            batch_size : int, default=128
                Batch size to use for training

            learning_rate : float, default=0.01
                Learning rate to use for training

            optimizer : optim.Optimizer, default=torch.optim.SGD
                Optimizer to use for training

            teach_ratio : float, default=0.5
                Ratio of sequences to train including labels

            verbose : boolean, default=True
                If True, prints progress

            Returns
            -------
            result : torch.Tensor
                Predictions corresponding to X
            """
        logger.info("fit_predict {} samples".format(X.shape[0]))

        # Apply fit and predict in sequence
        return self.fit(
            X             = X,
            y             = y,
            epochs        = epochs,
            batch_size    = batch_size,
            learning_rate = learning_rate,
            optimizer     = optimizer,
            teach_ratio   = teach_ratio,
            verbose       = verbose,
        ).predict(X)

    ########################################################################
    #                         ContextBuilder Query                         #
    ########################################################################

    def query(self, X, y, iterations=0, batch_size=1024, ignore=None,
              return_optimization=None, verbose=True):
        """Query the network to get optimal attention vector.

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context of events, same as input to fit and predict

            y : array-like of type=int and shape=(n_samples,)
                Observed event

            iterations : int, default=0
                Number of iterations to perform for optimization of actual event

            batch_size : int, default=1024
                Batch size of items to optimize

            ignore : int, optional
                If given ignore this index as attention

            return_optimization : float, optional
                If given, returns number of items with confidence level larger
                than given parameter. E.g. return_optimization=0.2 will also
                return two boolean tensors for elements with a confidence >= 0.2
                before optimization and after optimization.

            verbose : boolean, default=True
                If True, print progress

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, output_size)
                Confidence of each prediction given new attention

            attention : torch.Tensor of shape=(n_samples, context_size)
                Importance of each input with respect to output

            inverse : torch.Tensor of shape=(n_samples,)
                Inverse is returned to reconstruct the original array

            confidence_orig : torch.Tensor of shape=(n_samples,)
                Only returned if return_optimization != None
                Boolean array of items >= threshold before optimization

            confidence_optim : torch.Tensor of shape=(n_samples,)
                Only returned if return_optimization != None
                Boolean array of items >= threshold after optimization
            """
        # Get device
        original_device = X.device

        # Initialise result
        result_confidence = list()
        result_attention  = list()

        # Memory optimization, only use unique values
        X, y, inverse = unique_2d(X, y)

        # Ignore given datapoints
        if ignore is not None:
            raise NotImplementedError("Ignore is not properly implemented yet.")
            attention[X == ignore] = 0

        # Squeeze variables
        y = y.squeeze(1)

        # Initialise progress if necessary
        if verbose:
            progress = tqdm(None,
                total = int(iterations)*int(math.ceil(X.shape[0]/batch_size)),
                desc  = "Optimizing query",
            )

        # Batch data
        batches = DataLoader(
            TensorDataset(X, y),
            batch_size = batch_size,
            shuffle    = False,
        )

        # Count datapoints with confidence >= 0.2
        if return_optimization is not None:
            confidence_orig  = list()
            confidence_optim = list()

        ################################################################
        #                    Attention optimisation                    #
        ################################################################

        # Loop over batches
        for batch, (X_, y_) in enumerate(batches):
            # Compute initial attention and confidence
            confidence, attention = self.predict(X_, y_)
            confidence = confidence.squeeze(1)
            attention  = attention .squeeze(1)

            # Count confidence >= 0.2 of non-optimized datapoints
            if return_optimization is not None:
                confidence_orig.append((
                    confidence[torch.arange(y_.shape[0]), y_].exp() >= return_optimization
                ).detach().clone())

            # Make attention variable
            attn = Variable(attention.detach().clone(), requires_grad=True)
            # Set optimizer
            optimizer = optim.Adam([attn], lr=0.1)
            criterion = nn.NLLLoss()

            # Encode values of X
            with torch.no_grad():
                X_, _ = self.encoder(X_)

            # Perform iterations
            for iteration in range(int(iterations)):
                # Clear optimizer
                optimizer.zero_grad()

                # Add decoding function
                def decode(input, attn, softmax=False):
                    if softmax: attn = F.softmax(attn, dim=1)
                    return self.decoder_event(input, attn)

                # Perform prediction
                pred = decode(X_, attn, softmax=iteration > 0)

                # Compute loss
                loss = criterion(pred, y_)

                # Perform backpropagation
                loss.backward()
                optimizer.step()

                # Update progress if necessary
                if verbose: progress.update()

            # Perform final softmax
            if iterations > 0: attn = F.softmax(attn, dim=1)

            # Detach attention - memory optimization
            attn = attn.detach()

            # Get confidence levels
            confidence_ = self.decoder_event(X_, attn)
            confidence_ = confidence_[torch.arange(y_.shape[0]), y_].exp().detach()
            confidence  = confidence [torch.arange(y_.shape[0]), y_].exp().detach()

            # Check where confidence improved
            mask = confidence_ > confidence

            # Store attention if we improved
            attention[mask] = attn[mask]

            # Recompute confidence
            with torch.no_grad():
                confidence = self.decoder_event(
                    X         = X_,
                    attention = attention,
                ).exp()

                # Count confidence >= 0.2 of optimized datapoints
                if return_optimization is not None:
                    confidence_optim.append((
                        confidence[torch.arange(y_.shape[0]), y_] >= return_optimization
                    ).detach().clone())

            # Add confidence and attention to result
            result_confidence.append(confidence.cpu())
            result_attention .append(attention .cpu())

        # Combine confidence and attention into tensor
        # and cast to original device
        confidence = torch.cat(result_confidence).to(original_device)
        attention  = torch.cat(result_attention) .to(original_device)

        # Close progress if necessary
        if verbose: progress.close()

        # Return confidence optimization if necessary
        if return_optimization is not None:
            confidence_orig  = torch.cat(confidence_orig )
            confidence_optim = torch.cat(confidence_optim)
            # Return result
            return confidence, attention, inverse, confidence_orig, confidence_optim

        # Return result
        return confidence, attention, inverse


    ########################################################################
    #                           Save/load model                            #
    ########################################################################

    def save(self, outfile):
        """Save model to output file.

            Parameters
            ----------
            outfile : string
                File to output model.
            """
        # Save to output file
        torch.save(self.state_dict(), outfile)

    @classmethod
    def load(cls, infile, device=None):
        """Load model from input file.

            Parameters
            ----------
            infile : string
                File from which to load model.
            """
        # Load state dictionary
        state_dict = torch.load(infile, map_location=device)

        # Get input variables from state_dict
        input_size    = state_dict.get('embedding.weight').shape[0]
        output_size   = state_dict.get('decoder_event.out.weight').shape[0]
        hidden_size   = state_dict.get('embedding.weight').shape[1]
        num_layers    = 1 # TODO
        max_length    = state_dict.get('decoder_attention.attn.weight').shape[0]
        bidirectional = state_dict.get('decoder_attention.attn.weight').shape[1] // hidden_size != num_layers
        LSTM          = False # TODO

        # Create ContextBuilder
        result = cls(
            input_size    = input_size,
            output_size   = output_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            max_length    = max_length,
            bidirectional = bidirectional,
            LSTM          = LSTM,
        )

        # Cast to device if necessary
        if device is not None: result = result.to(device)

        # Set trained parameters
        result.load_state_dict(state_dict)

        # Return result
        return result
