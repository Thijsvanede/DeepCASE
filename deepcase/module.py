# Imports
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom imports
from deepcase.context_builder       import ContextBuilder
from deepcase.interpreter           import Interpreter
from deepcase.context_builder.utils import unique_2d

# Set logger
logger = logging.getLogger(__name__)

class DeepCASE(object):

    def __init__(self,
            n_features,
            max_length  = 10,
            hidden_size = 128,
            eps         = 0.1,
            min_samples = 5,
            threshold   = 0.2,
        ):
        """Analyse security events with respect to contextual machine behaviour.

            Note
            ----
            When an Interpreter is trained, it heavily depends on the
            ContextBuilder used during training. Therefore, we **strongly**
            suggest **not** to manually change the context_builder attribute,
            without retraining the interpreter of the DeepCASE object.

            Parameters
            ----------
            n_features : int
                Number of different possible security events.

            max_length : int, default=10
                Maximum length of context window as number of events.

            hidden_size : int, default=128
                Size of hidden layer in sequence to sequence prediction.
                This parameter determines the complexity of the model and its
                prediction power. However, high values will result in slower
                training and prediction times.

            eps : float, default=0.1
                Epsilon used for determining maximum distance between clusters.

            min_samples : int, default=5
                Minimum number of required samples per cluster.

            threshold : float, default=0.2
                Minimum required confidence in fingerprint before using it in
                training clusters.
            """

        # Initialise Context Builder from parameters
        self.context_builder = ContextBuilder(
            input_size    = n_features,
            output_size   = n_features,
            max_length    = max_length,
            hidden_size   = hidden_size,
        )

        # Initialise Interpreter from parameters
        self.interpreter = Interpreter(
            context_builder = self.context_builder,
            features        = n_features,
            eps             = eps,
            min_samples     = min_samples,
            threshold       = threshold,
        )

    ########################################################################
    #                              Fit method                              #
    ########################################################################

    def fit(self,
            # Input data
            X,
            y,
            scores,

            # ContextBuilder-specific parameters
            epochs        = 10,
            batch_size    = 128,
            learning_rate = 0.01,
            optimizer     = optim.SGD,
            teach_ratio   = 0.5,

            # Interpreter-specific parameters
            iterations       = 100,
            query_batch_size = 1024,
            strategy         = "max",
            NO_SCORE         = -1,

            # Verbosity level
            verbose = True,
        ):
        """Fit DeepCASE with given data.
            This method is provided as a wrapper and is equivalent to calling:
             - context_builder.fit() and
             - interpreter.fit()
            in the given order.

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context to train with.

            y : array-like of type=int and shape=(n_samples, n_future_events)
                Sequences of target events.

            scores : array-like of float, shape=(n_samples,)
                Scores for each sample in cluster.

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

            iterations : int, default=100
                Number of iterations for query.

            query_batch_size : int, default=1024
                Size of batch for query.

            strategy : string (max|min|avg), default=max
                Strategy to use for computing scores per cluster based on scores
                of individual events. Currently available options are:
                - max: Use maximum score of any individual event in a cluster.
                - min: Use minimum score of any individual event in a cluster.
                - avg: Use average score of any individual event in a cluster.

            NO_SCORE : float, default=-1
                Score to indicate that no score was given to a sample and that
                the value should be ignored for computing the cluster score.
                The NO_SCORE value will also be given to samples that do not
                belong to a cluster.

            verbose : boolean, default=True
                If True, prints progress.

            Returns
            -------
            self : self
                Returns self.
            """

        # Fit the ContextBuilder
        self.context_builder.fit(
            X             = X,
            y             = y,
            epochs        = epochs,
            batch_size    = batch_size,
            learning_rate = learning_rate,
            optimizer     = optimizer,
            teach_ratio   = teach_ratio,
            verbose       = verbose,
        )

        # Fit the Interpreter
        self.interpreter.fit(
            X          = X,
            y          = y,
            scores     = scores,
            iterations = iterations,
            batch_size = query_batch_size,
            strategy   = strategy,
            NO_SCORE   = NO_SCORE,
            verbose    = verbose,
        )

        # Returns self
        return self


    ########################################################################
    #                            Predict method                            #
    ########################################################################

    def predict(self, X, y, iterations=100, batch_size=1024, verbose=False):
        """Predict maliciousness of context samples.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context for which to predict maliciousness.

            y : torch.Tensor of shape=(n_samples, 1)
                Events for which to predict maliciousness.

            iterations : int, default=100
                Iterations used for optimization.

            batch_size : int, default=1024
                Batch size used for optimization.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Predicted maliciousness score.
                Positive scores are maliciousness scores.
                A score of 0 means we found a match that was not malicious.
                Special cases:

                * -1: Not confident enough for prediction
                * -2: Label not in training
                * -3: Closest cluster > epsilon
            """
        # Return the prediction of the interpreter
        return self.interpreter.predict(
            X          = X,
            y          = y,
            iterations = iterations,
            batch_size = batch_size,
            verbose    = verbose,
        )

    ########################################################################
    #                         Fit/predict methods                          #
    ########################################################################

    def fit_predict(self,
            # Input data
            X,
            y,
            scores,

            # ContextBuilder-specific parameters
            epochs        = 10,
            batch_size    = 128,
            learning_rate = 0.01,
            optimizer     = optim.SGD,
            teach_ratio   = 0.5,

            # Interpreter-specific parameters
            iterations       = 100,
            query_batch_size = 1024,
            strategy         = "max",
            NO_SCORE         = -1,

            # Verbosity level
            verbose = True,
        ):
        """Fit DeepCASE with given data and predict that same data.
            This method is provided as a wrapper and is equivalent to calling:
             - self.fit() and
             - self.predict()
            in the given order.

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context to train with.

            y : array-like of type=int and shape=(n_samples, n_future_events)
                Sequences of target events.

            scores : array-like of float, shape=(n_samples,)
                Scores for each sample in cluster.

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

            iterations : int, default=100
                Number of iterations for query.

            query_batch_size : int, default=1024
                Size of batch for query.

            strategy : string (max|min|avg), default=max
                Strategy to use for computing scores per cluster based on scores
                of individual events. Currently available options are:
                - max: Use maximum score of any individual event in a cluster.
                - min: Use minimum score of any individual event in a cluster.
                - avg: Use average score of any individual event in a cluster.

            NO_SCORE : float, default=-1
                Score to indicate that no score was given to a sample and that
                the value should be ignored for computing the cluster score.
                The NO_SCORE value will also be given to samples that do not
                belong to a cluster.

            verbose : boolean, default=True
                If True, prints progress.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Predicted maliciousness score.
                Positive scores are maliciousness scores.
                A score of 0 means we found a match that was not malicious.
                Special cases:

                * -1: Not confident enough for prediction
                * -2: Label not in training
                * -3: Closest cluster > epsilon
            """
        # Call fit and predict in sequence and return result
        return self.fit(
            X                = X,
            y                = y,
            scores           = scores,
            epochs           = epochs,
            batch_size       = batch_size,
            learning_rate    = learning_rate,
            optimizer        = optimizer,
            teach_ratio      = teach_ratio,
            iterations       = iterations,
            query_batch_size = query_batch_size,
            strategy         = strategy,
            NO_SCORE         = NO_SCORE,
            verbose          = verbose,
        ).predict(
            X          = X,
            y          = y,
            iterations = iterations,
            batch_size = query_batch_size,
            verbose    = verbose,
        )

    ########################################################################
    #                            Cast to device                            #
    ########################################################################

    def to(self, device):
        """Cast DeepCASE to a specific device.
            This method acts as a wrapper for the underlying context_builder.

            Parameters
            ----------
            device : string
                String describing the device, e.g., "cpu", "cuda", or "cuda:0".

            Returns
            -------
            self : self
                Returns self
            """
        # Cast ContextBuilder to device
        self.context_builder = self.context_builder.to(device)

        # Return self
        return self

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    # def save(self, context_builder=None, interpreter=None):
    #     """Save the DeepCASE object to output files.
    #
    #         Parameters
    #         ----------
    #         context_builder : string, optional
    #             If given, output ContextBuilder to given file.
    #
    #         interpreter : string, optional
    #             If given, output Interpreter to given file.
    #         """
    #     logger.info("DeepCASE.save(context_builder={}, interpreter={})".format(
    #         context_builder, interpreter))
    #
    #     # Save individual objects
    #     if context_builder is not None:
    #         self.context_builder.save(context_builder)
    #     if interpreter is not None:
    #         self.interpreter.save(interpreter)
    #
    # @classmethod
    # def load(cls, context_builder=None, interpreter=None):
    #     logger.info("DeepCASE.load(context_builder={}, interpreter={})".format(
    #         context_builder, interpreter))
    #
    #     # Load ContextBuilder
    #     if context_builder is not None:
    #         context_builder = ContextBuilder.load(context_builder)
    #         print(context_builder)
    #     # Load Interpreter
    #     if interpreter is not None:
    #         interpreter = Interpreter.load(interpreter, context_builder)
    #         print(interpreter)
