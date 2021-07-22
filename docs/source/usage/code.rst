Code integration
================
To integrate DeepCASE into your own project, you can use it as a standalone module.
DeepCASE offers rich functionality that is easy to integrate into other projects.
Here we show some simple examples on how to use the DeepCASE package in your own python code.
For a complete documentation we refer to the :ref:`Reference` guide.

.. Note::

   The code used in this section is also available in the GitHub repository under ``examples/example.py``.

Import
^^^^^^
To import components from DeepCASE simply use the following format

.. code:: python

   from deepcase(.<module>) import <Object>

For example, the following code imports the :ref:`Preprocessor`, :ref:`ContextBuilder`, and :ref:`Interpreter`.

.. code:: python

   from deepcase.preprocessing   import Preprocessor
   from deepcase.context_builder import ContextBuilder
   from deepcase.interpreter     import Interpreter

Loading data
^^^^^^^^^^^^
DeepCASE can load sequences from ``.csv`` and specifically formatted ``.txt`` files (see :ref:`Preprocessor` class).

.. code:: python

    # Create preprocessor
    preprocessor = Preprocessor(
        length  = 10,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
    )

    # Load data from file
    context, events, labels, mapping = preprocessor.csv('data/example.csv')

In case no labels were explicitly provided as an argument, and no labels could be extracted from the file, we may set labels for each sequence manually.
Note that we assign the labels as a numpy array, which requires importing numpy using ``import numpy as np``.

.. code:: python

    # In case no labels are provided, set labels to -1
    if label is None:
        labels = np.full(events.shape[0], -1, dtype=int)

By default, the Tensors returned by the :ref:`Preprocessor` are set to the ``cpu`` device.
If you have a system that supports ``cuda`` Tensors you can cast the Tensors to cuda using the following code.
Note that the check in this code requires you to import PyTorch using ``import torch``.

.. code:: python

    # Cast to cuda if available
    if torch.cuda.is_available():
        events  = events .to('cuda')
        context = context.to('cuda')

Splitting data
--------------
Once we have loaded the data, we will split it into train and test data.
This step is not necessarily required, depending on the setup you use, but we will use the training and test data in the remainder of this example.

.. code:: python

    # Split into train and test sets (20:80) by time - assuming events are ordered chronologically
    events_train  = events [:events.shape[0]//5 ]
    events_test   = events [ events.shape[0]//5:]

    context_train = context[:events.shape[0]//5 ]
    context_test  = context[ events.shape[0]//5:]

    label_train   = label  [:events.shape[0]//5 ]
    label_test    = label  [ events.shape[0]//5:]

ContextBuilder
^^^^^^^^^^^^^^
First we create an instance of DeepCASE's :ref:`ContextBuilder` using the following code:

.. code:: python

    # Create ContextBuilder
    context_builder = ContextBuilder(
        input_size    = 100,   # Number of input features to expect
        output_size   = 100,   # Same as input size
        hidden_size   = 128,   # Number of nodes in hidden layer, in paper we set this to 128
        max_length    = 10,    # Length of the context, should be same as context in Preprocessor
    )

    # Cast to cuda if available
    if torch.cuda.is_available():
        context_builder = context_builder.to('cuda')

Once the ``context_builder`` is created, we train it using the :py:meth:`fit()` method.

.. code:: python

    # Train the ContextBuilder
    context_builder.fit(
        X             = context_train,               # Context to train with
        y             = events_train.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
        epochs        = 10,                          # Number of epochs to train with
        batch_size    = 128,                         # Number of samples in each training batch, in paper this was 128
        learning_rate = 0.01,                        # Learning rate to train with, in paper this was 0.01
        verbose       = True,                        # If True, prints progress
    )

I/O methods
-----------
We can load and save the ContextBuilder to and from a file using the following code:

.. code:: python

    # Save ContextBuilder to file
    context_builder.save('path/to/file.save')
    # Load ContextBuilder from file
    context_builder = ContextBuilder.load('path/to/file.save')

Interpreter
^^^^^^^^^^^
Once we fitted the ``context_builder``, we create in :ref:`Interpreter` instance using the following code:

.. code:: python

    # Create Interpreter
    interpreter = Interpreter(
        context_builder = context_builder, # ContextBuilder used to fit data
        features        = 100,             # Number of input features to expect, should be same as ContextBuilder
        eps             = 0.1,             # Epsilon value to use for DBSCAN clustering, in paper this was 0.1
        min_samples     = 5,               # Minimum number of samples to use for DBSCAN clustering, in paper this was 5
        threshold       = 0.2,             # Confidence threshold used for determining if attention from the ContextBuilder can be used, in paper this was 0.2
    )

Once the ``interpreter`` is created, we can use it to cluster samples using the :py:meth:`cluster()` method.

.. code:: python

    # Cluster samples with the interpreter
    clusters = interpreter.cluster(
        X          = context_train,               # Context to train with
        y          = events_train.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
        iterations = 100,                         # Number of iterations to use for attention query, in paper this was 100
        batch_size = 1024,                        # Batch size to use for attention query, used to limit CUDA memory usage
        verbose    = True,                        # If True, prints progress
    )

I/O methods
-----------
We can load and save the Interpreter to and from a file using the following code:

.. code:: python

    # Save Interpreter to file
    interpreter.save('path/to/file.save')
    # Load Interpreter from file
    interpreter = Interpreter.load(
        'path/to/file.save',
        context_builder = context_builder, # When loading the Interpreter, make sure it is linked to the same ContextBuilder used for training.
    )

Manual Mode
^^^^^^^^^^^
When we have used the Interpreter to cluster samples, we can assign a score to the individual clusters.
Assigning a score is done through the :py:meth:`score()` method, however, this method has two requirements for assigning a score:

1. that all sequences used to create clusters are assigned a score.
2. that all sequences in the **same** cluster are assigned the **same** score.

Therefore, to make sure these two conditions hold, we first call the :py:meth:`score_clusters()` method and use the result for the :py:meth:`score()` method.

.. code:: python

    # Compute scores for each cluster based on individual labels per sequence
    scores = interpreter.score_clusters(
        scores   = labels_train, # Labels used to compute score (either as loaded by Preprocessor, or put your own labels here)
        strategy = "max",        # Strategy to use for scoring (one of "max", "min", "avg")
        NO_SCORE = -1,           # Any sequence with this score will be ignored in the strategy.
                                 # If assigned a cluster, the sequence will inherit the cluster score.
                                 # If the sequence is not present in a cluster, it will receive a score of NO_SCORE.
    )

    # Assign scores to clusters in interpreter
    # Note that all sequences should be given a score and each sequence in the
    # same cluster should have the same score.
    interpreter.score(
        scores  = scores, # Scores to assign to sequences
        verbose = True,   # If True, prints progress
    )

Semi-automatic Mode
^^^^^^^^^^^^^^^^^^^
Once we used the :ref:`Interpreter` for clustering and assigned a score to each cluster, we can use the :py:meth`predict()` method to predict labels of new sequences.
When no cluster could be matched, the :py:meth`predict()` method gives one of three scores for a cluster:

 * ``-1``, if the :ref:`ContextBuilder` is not confident enough for a prediction.
 * ``-2``, if the ``event`` was not in the training dataset.
 * ``-3``, if the nearest cluster is a larger distance than ``epsilon`` away from the nearest sequence.

.. code:: python

    # Compute predicted scores
    prediction = interpreter.predict(
        X          = context_test,               # Context to predict
        y          = events_test.reshape(-1, 1), # Events to predict, note that these should be of shape=(n_events, 1)
        iterations = 100,                        # Number of iterations to use for attention query, in paper this was 100
        batch_size = 1024,                       # Batch size to use for attention query, used to limit CUDA memory usage
        verbose    = True,                       # If True, prints progress
    )
