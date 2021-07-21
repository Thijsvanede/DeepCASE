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

For example, the following code imports :ref:`DeepCASE`, the :ref:`Preprocessor`, :ref:`ContextBuilder`, and :ref:`Interpreter`.

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
TODO
