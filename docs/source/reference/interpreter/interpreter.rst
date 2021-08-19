.. _Interpreter:

Interpreter
===========
The Interpreter takes input sequences (``context`` and ``events``) and clusters them.
In order to do this clustering, it uses the ``attention`` values from the :ref:`ContextBuilder` after applying the attention query.
Besides clustering, the Interpreter also offers methods to assign scores for Manual analysis, and to predict the scores of unknown sequences for Semi-Automatic analysis.

.. autoclass:: interpreter.Interpreter

.. automethod:: interpreter.Interpreter.__init__

Fit/Predict methods
^^^^^^^^^^^^^^^^^^^
We provide a ``scikit-learn``-like API for the Interpreter as a classifier to labels for sequences in the form of clusters and predict the labels of new sequences.
To this end, we implement scikit-learn like fit and predict methods for training and predicting with the network.

Fit
~~~
The :py:meth:`fit()` method provides an API for directly learning the maliciousness score of sequences.
This method combines Interpreter's Clustering and Manual Mode for sequences where the labels are known a priori.
To this end, it calls the :py:meth:`cluster()`, :py:meth:`score_clusters()`, and :py:meth:`score()` methods in sequence.
When the labels for sequences are not known in advance, the Interpreter offers the functionality to first cluster sequences, and then manually inspect clusters for labelling as described in the paper.
For this functionality, we refer to the methods:

 * :py:meth:`interpreter.Interpreter.cluster()`
 * :py:meth:`interpreter.Interpreter.score_clusters()`
 * :py:meth:`interpreter.Interpreter.score()`

.. automethod:: interpreter.Interpreter.fit

Predict
~~~~~~~
When the Interpreter is trained using either the :py:meth:`fit()` method, or by using the individual :py:meth:`cluster()` and :py:meth:`score()` methods, we can use the Interpreter in (semi-)automatic mode.
To this end, we provide the :py:meth:`predict()` function which takes ``context`` and ``events`` as input and outputs the labels of corresponding predicted clusters.
If no sequence could be matched, one of the following scores will be given:

 * ``-1``: Not confident enough for prediction
 * ``-2``: Label not in training
 * ``-3``: Closest cluster > epsilon

.. Note::

    To use the :py:meth:`predict()` method, make sure that **both** the :py:meth:`cluster()` and :py:meth:`score()` methods have been called to cluster samples and assign a score to those samples.

.. automethod:: interpreter.Interpreter.predict

Fit_predict
~~~~~~~~~~~
Similar to the ``scikit-learn`` API, the :py:meth:`fit_predict()` method performs the :py:meth:`fit()` and :py:meth:`predict()` functions in sequence on the same data.

.. automethod:: interpreter.Interpreter.fit_predict


Clustering
^^^^^^^^^^
The main task of the Interpreter is to cluster events.
To this end, the :py:meth:`cluster()` method automatically clusters sequences from the ``context`` and ``events`` that have been given as input.

.. automethod:: interpreter.Interpreter.cluster

Auxiliary cluster methods
~~~~~~~~~~~~~~~~~~~~~~~~~
To create clusters, we recall from the DeepCASE paper Section III-C1 we apply :py:meth:`attention_query()` to the result from the ContxtBuilder.
Using the obtained attention we create a vector (Section III-B2) representing the context using the method :py:meth:`vectorize()`.
Both steps are combined in the method :py:meth:`attended_context()`.

.. automethod:: interpreter.Interpreter.attended_context

.. automethod:: interpreter.Interpreter.attention_query

.. automethod:: interpreter.Interpreter.vectorize


Manual mode
^^^^^^^^^^^
Once events have been clusters, we can assign a label or score to each sequence.
This way, we manually label the clusters and prepare the Interpreter object for (semi-)automatically predicting labels for new sequences.
To assign labels to clusters, we provide the :py:meth:`score()` method.

.. Note::

   The :py:meth:`score()` function requires:
    1. that all sequences used to create clusters are assigned a score.
    2. that all sequences in the **same** cluster are assigned the **same** score.

   If you do not have labels for all clusters or different labels within the same cluster, the :py:meth:`interpreter.Interpreter.score_clusters()` method prepares scores such that both conditions are satisfied.

.. automethod:: interpreter.Interpreter.score

Auxiliary manual methods
~~~~~~~~~~~~~~~~~~~~~~~~
As mentioned above, the :py:meth:`score()` function has two requirements:
 1. that all sequences used to create clusters are assigned a score.
 2. that all sequences in the **same** cluster are assigned the **same** score.

We provide the :py:meth:`score_clusters()` method for the situations where you only have labels for some sequences, or if the labels for sequences within the same cluster are not necessarily equal.
This method will apply a given ``strategy`` for equalizing the labels per cluster.
Additionally, unlabelled clusters will all be labeled using a given ``NO_SCORE`` score.

.. automethod:: interpreter.Interpreter.score_clusters

Semi-automatic mode
^^^^^^^^^^^^^^^^^^^
See :py:meth:`interpreter.Interpreter.predict()`.

I/O methods
^^^^^^^^^^^
The Interpreter can be saved and loaded from files using the following methods.
Please note that the :py:meth:`interpreter.Interpreter.load()` method is a ``classmethod`` and must be called statically.

.. automethod:: interpreter.Interpreter.save

.. automethod:: interpreter.Interpreter.load

**Example:**

.. code:: python

   from deepcase.interpreter import Interpreter
   interpreter = Interpreter.load('<path_to_saved_interpreter>')
   interpreter.save('<path_to_save_interpreter>')
