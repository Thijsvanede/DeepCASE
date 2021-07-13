.. _Interpreter:

Interpreter
===========

.. Note::

   Currently, only the fit/predict methods are documented.
   The methods used for extracting a 'fingerprint' from sequences and clustering different sequences are likely to change in the next iteration release.
   The fit/predict methods may also change to better reflect the DeepCASE design.
   To this end, we will split the cluster creation and labelling into separate methods.


The Interpreter takes input sequences, extracts the relevant items in the input sequences using using the :ref:`ContextBuilder`, and clusters these according to similarity.

.. autoclass:: interpreter.Interpreter

.. automethod:: interpreter.Interpreter.__init__

Class methods
^^^^^^^^^^^^^

Fit/Predict methods
-------------------
We provide the Interpreter as a clustering mechanism to group similar sequences together.
To this end, we implement `scikit-learn <https://scikit-learn.org/stable/>`_ like fit and predict methods for training and predicting sequences.

The ``fit()`` method automatically clusters security event sequences using the given input data ``X`` and ``y``.
It labels the produced clusters with the maximum score of an individual sequence within that cluster.
Please see the method below for all available options.

.. automethod:: interpreter.Interpreter.fit

The ``predict()`` method predicts the ``maliciousness`` score for matched sequences.
If no sequence could be matched, one of the following scores will be given:

* ``-1``: Not confident enough for prediction
* ``-2``: Label not in training
* ``-3``: Closest cluster > epsilon

.. automethod:: interpreter.Interpreter.predict

The ``fit_predict()`` method performs the ``fit()`` and ``predict()`` functions in sequence on the same data.

.. automethod:: interpreter.Interpreter.fit_predict
