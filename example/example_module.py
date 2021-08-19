# Other imports
import numpy as np
import torch

# DeepCASE Imports
from deepcase.preprocessing import Preprocessor
from deepcase               import DeepCASE

if __name__ == "__main__":
    ########################################################################
    #                             Loading data                             #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        length  = 10,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
    )

    # Load data from file
    context, events, labels, mapping = preprocessor.csv('data/example.csv')

    # In case no labels are provided, set labels to -1
    if labels is None:
        labels = np.full(events.shape[0], -1, dtype=int)

    # Cast to cuda if available
    if torch.cuda.is_available():
        events  = events .to('cuda')
        context = context.to('cuda')

    ########################################################################
    #                            Splitting data                            #
    ########################################################################

    # Split into train and test sets (20:80) by time - assuming events are ordered chronologically
    events_train  = events [:events.shape[0]//5 ]
    events_test   = events [ events.shape[0]//5:]

    context_train = context[:events.shape[0]//5 ]
    context_test  = context[ events.shape[0]//5:]

    labels_train  = labels [:events.shape[0]//5 ]
    labels_test   = labels [ events.shape[0]//5:]

    ########################################################################
    #                            Using DeepCASE                            #
    ########################################################################

    deepcase = DeepCASE(
        # ContextBuilder parameters
        features    = 300, # Number of input features to expect
        max_length  = 10,  # Length of the context, should be same as context in Preprocessor
        hidden_size = 128, # Number of nodes in hidden layer, in paper we set this to 128

        # Interpreter parameters
        eps         = 0.1, # Epsilon value to use for DBSCAN clustering, in paper this was 0.1
        min_samples = 5,   # Minimum number of samples to use for DBSCAN clustering, in paper this was 5
        threshold   = 0.2, # Confidence threshold used for determining if attention from the ContextBuilder can be used, in paper this was 0.2
    )

    # Cast to cuda if available
    if torch.cuda.is_available():
        deepcase = deepcase.to('cuda')

    ########################################################################
    #                             Fit DeepCASE                             #
    ########################################################################

    # Train the ContextBuilder
    # Conveniently, the fit and fit_predict methods have the same API, so if you
    # do not require the predicted values on the train dataset, simply
    # substitute fit_predict with fit and it will run slightly quicker because
    # DeepCASE skip the prediction over the training dataset and simply return
    # the deepcase object itself. Other than that, both calls are exactly the
    # same.
    prediction_train = deepcase.fit_predict(
        # Input data
        X      = context_train,               # Context to train with
        y      = events_train.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
        scores = labels_train,                # Labels used to compute score (either as loaded by Preprocessor, or put your own labels here)

        # ContextBuilder-specific parameters
        epochs        = 10,                   # Number of epochs to train with
        batch_size    = 128,                  # Number of samples in each training batch, in paper this was 128
        learning_rate = 0.01,                 # Learning rate to train with, in paper this was 0.01

        # Interpreter-specific parameters
        iterations       = 100,               # Number of iterations to use for attention query, in paper this was 100
        query_batch_size = 1024,              # Batch size to use for attention query, used to limit CUDA memory usage
        strategy         = "max",             # Strategy to use for scoring (one of "max", "min", "avg")
        NO_SCORE         = -1,                # Any sequence with this score will be ignored in the strategy.
                                              # If assigned a cluster, the sequence will inherit the cluster score.
                                              # If the sequence is not present in a cluster, it will receive a score of NO_SCORE.

        # Verbosity level
        verbose = True,                       # If True, prints progress
    )

    ########################################################################
    #                        Predict with DeepCASE                         #
    ########################################################################

    # Compute predicted scores
    prediction_test = deepcase.predict(
        X          = context_test,               # Context to predict
        y          = events_test.reshape(-1, 1), # Events to predict, note that these should be of shape=(n_events, 1)
        iterations = 100,                        # Number of iterations to use for attention query, in paper this was 100
        batch_size = 1024,                       # Batch size to use for attention query, used to limit CUDA memory usage
        verbose    = True,                       # If True, prints progress
    )
