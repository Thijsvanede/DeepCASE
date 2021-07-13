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

   from deepcase                 import DeepCASE
   from deepcase.preprocessing   import Preprocessor
   from deepcase.context_builder import ContextBuilder
   from deepcase.interpreter     import Interpreter

Loading data
^^^^^^^^^^^^
DeepCASE can load sequences from ``.csv`` and specifically formatted ``.txt`` files (see :ref:`Preprocessor` class).

.. code:: python

    # Create preprocessor
    preprocessor = Preprocessor(
        context = 10,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
    )

    # Load data from file
    events, context, label, mapping = preprocessor.csv('data/atlas.csv')

In case no labels were explicitly provided as an argument, and no labels could be extracted from the file, we may set the labels manually.

.. code:: python

    # In case no labels are provided, set labels to 0
    if label is None:
        label = torch.zeros(events.shape[0], dtype = torch.long)

By default, the Tensors returned by the :ref:`Preprocessor` are set to the ``cpu`` device.
If you have a system that supports ``cuda`` Tensors you can cast the Tensors to cuda using the following code.
Note that the check in this code requires you to import PyTorch using ``import torch``.

.. code:: python

    # Cast to cuda if available
    if torch.cuda.is_available():
        events  = events .to('cuda')
        context = context.to('cuda')
        label   = label  .to('cuda')

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

Using DeepCASE
^^^^^^^^^^^^^^
First we create an instance of DeepCASE using the following code.

.. code:: python

    # Create DeepCASE object
    deepcase = DeepCASE(
        n_features  = 60,                                             # Set maximum number of expected events (60 is enough for the ATLAS dataset)
        complexity  = 128,                                            # Default complexity used in DeepCASE, dimension of hidden layer
        context     = 10,                                             # 10 events in context, same as in preprocessor
        device      = 'cuda' if torch.cuda.is_available() else 'cpu', # Or manually set 'cpu'/'cuda'
        eps         = 0.1,                                            # Default epsilon     used in DeepCASE, used for DBSCAN clustering
        min_samples = 5,                                              # Default min_samples used in DeepCASE, used for DBSCAN clustering
        threshold   = 0.2,                                            # Default threshold   used in DeepCASE, minimum required confidence
    )

Once the object is created, we can access the :ref:`ContextBuilder` and :ref:`Interpreter` as follows

.. code:: python

    deepcase.context_builder # Returns the ContextBuilder object
    deepcase.interpreter     # Returns the Interpreter    object

Training DeepCASE - Manual Mode
-------------------------------
We can now feed the training data to DeepCASE.
To do so, we first train the ContextBuilder on the training data.
Next we use the Interpreter to create (and possibly label) clusters.

.. code:: python

    # Fit ContextBuilder
    deepcase.context_builder.fit(
        X          = context_train,
        y          = events_train,
        batch_size = 128,           # Batch size you want to train with
        epochs     = 10,            # Number of epochs to train
        verbose    = True,          # If True, prints training progress
    )

    # Fit Interpreter
    deepcase.interpreter.fit(
        X          = context_train,
        y          = events_train,
        score      = label_train.unsqueeze(1),
        verbose    = True,
    )

Once we fit the :ref:`Interpreter` with the training data, we can inspect the clusters for each input using ``deepcase.interpreter.clusters``.

Running DeepCASE - Semi-automatic mode
--------------------------------------
Once DeepCASE is trained with the known labels, we can match new sequences against known clusters.
We do this using the :ref:`Interpreter`'s ``predict()`` function.

.. code:: python

    # Use deepcase to predict labels
    label_predict = deepcase.interpreter.predict(
        X       = context_test,
        y       = events_test,
        verbose = True,
    )

This prediction returns an average score of the items in the matching cluster.
Hence, to get an exact prediction, you will need to manually set a policy to transform the returned scores into a risk level.

Save & Load components
----------------------
Training the ContextBuilder and Interpreter can take some time, and you may want use those trained versions later.
Therefore, DeepCASE provides methods to save and load the :ref:`ContextBuilder` and :ref:`Interpreter` to and from files.

To save the components, simply use the :py:meth:`save()` method:

.. code:: python

    # Save DeepCASE components
    deepcase.context_builder.save('context.save')     # Or specify a different filename
    deepcase.interpreter    .save('interpreter.save') # Or specify a different filename

Once you have saved the components, you can load the components from the files at any point:

.. code:: python

    # Load DeepCASE components
    deepcase.context_builder.load(
        infile = 'context.save',                                 # File from which to load ContextBuilder
        device = 'cuda' if torch.cuda.is_available() else 'cpu', # Or manually set 'cpu'/'cuda'
    )

    deepcase.interpreter.load(
        infile          = 'interpreter.save',       # File from which to load Interpreter
        context_builder = deepcase.context_builder, # Used to link Interpreter to ContextBuilder. IMPORTANT: an Interpreter is specific to a ContextBuilder, so using a different ContextBuilder than used for training the Interpreter may yield bad results.
    )
