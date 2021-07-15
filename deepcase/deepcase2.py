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

    def __init__(self, n_features, context=10, complexity=128, device=None,
                 eps=0.1, min_samples=5, threshold=0.2):
        """Analyse security events with respect to contextual machine behaviour

            Parameters
            ----------
            n_features : int
                Number of different possible security events

            context : int, default=10
                Maximum size of context window as number of events

            complexity : int, default=128
                Complexity of the network, describes the number of hidden
                dimensions used by the sequence prediction algorithm.

            device : 'cpu'|'cuda<i?>'|None, default=None
                Hardware device to use for computation.
                If None, fastest hardware is automatically inferred
                If 'cuda' a specific graphics card can be selected using
                cuda<i>, where <i> is the identifier of the graphics card, e.g.
                'cuda0' for card 0.

            eps : float, default=0.1
                Epsilon used for determining maximum distance between clusters.

            min_samples : int, default=5
                Minimum number of required samples per cluster.

            threshold : float, default=0.2
                Minimum required confidence in fingerprint before using it in
                training clusters.
            """
        logger.info("DeepCASE.__init__(n_features={}, context={}, complexity={}"
                     ", device={}, eps={}, min_samples={}, threshold={})"
                     .format(n_features, context, complexity, device, eps,
                     min_samples, threshold))

        # Set device to use for prediction
        if device is None:
            # Automatically infer device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialise Context Builder
        self.context_builder = ContextBuilder(
            input_size    = n_features, # Possible different security events
            output_size   = n_features, # Possible different security events
            max_length    = context,    # Define maximum context window
            hidden_size   = complexity, # Complexity of network (simple)
            num_layers    = 1,          # Complexity of network (simple)
            bidirectional = False,      # Complexity of network (simple)
            LSTM          = False,      # Complexity of network (GRU/simple)
        ).to(self.device)

        # Initialise Interpreter
        self.interpreter = Interpreter(
            context_builder = self.context_builder,
            features        = n_features,
            eps             = eps,
            min_samples     = min_samples,
            threshold       = threshold,
        )

        logger.info("DeepCASE successfully created")

    ########################################################################
    #                             Set methods                              #
    ########################################################################

    @property
    def context_builder(self):
        return self._context_builder

    @context_builder.setter
    def context_builder(self, context_builder):
        # Set context_builder
        self._context_builder = context_builder
        # Set context_builder of interpreter
        if hasattr(self, 'interpreter') and\
           hasattr(self.interpreter, 'context_builder'):
            self.interpreter.context_builder = context_builder

    ########################################################################
    #                              Fit method                              #
    ########################################################################

    # TODO

    ########################################################################
    #                            Predict method                            #
    ########################################################################

    # TODO

    ########################################################################
    #                         Fit/predict methods                          #
    ########################################################################

    # TODO

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def save(self, context_builder=None, interpreter=None):
        """Save the DeepCASE object to output files.

            Parameters
            ----------
            context_builder : string, optional
                If given, output ContextBuilder to given file.

            interpreter : string, optional
                If given, output Interpreter to given file.
            """
        logger.info("DeepCASE.save(context_builder={}, interpreter={})".format(
            context_builder, interpreter))

        # Save individual objects
        if context_builder is not None:
            self.context_builder.save(context_builder)
        if interpreter is not None:
            self.interpreter.save(interpreter)

    @classmethod
    def load(cls, context_builder=None, interpreter=None):
        logger.info("DeepCASE.load(context_builder={}, interpreter={})".format(
            context_builder, interpreter))

        # Load ContextBuilder
        if context_builder is not None:
            context_builder = ContextBuilder.load(context_builder)
            print(context_builder)
        # Load Interpreter
        if interpreter is not None:
            interpreter = Interpreter.load(interpreter, context_builder)
            print(interpreter)

        exit()
