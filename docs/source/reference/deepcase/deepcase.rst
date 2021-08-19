.. _DeepCASE:

DeepCASE
========
We provide the DeepCASE class as a wrapper around the :ref:`ContextBuilder` and :ref:`Interpreter`.
The DeepCASE class only implements the ``fit()/predict()`` methods which requires a priori knowledge of the sequence maliciousness score.
If you require a more fine-grained tuning of DeepCASE, e.g., when using the manual labelling mode, we recommend using the individual :ref:`ContextBuilder` and :ref:`Interpreter` objects as shown in the :ref:`Usage`.
These individual classes provide a richer API.

.. autoclass:: module.DeepCASE

.. automethod:: module.DeepCASE.__init__

Fit/Predict methods
^^^^^^^^^^^^^^^^^^^
We provide a ``scikit-learn``-like API for DeepCASE to train on sequences with a given maliciousness score and predict the maliciousness score of new sequences.

As DeepCASE is simply a wrapper around the :ref:`ContextBuilder` and :ref:`Interpreter` objects, the following functionality is equivalent:

:py:meth:`module.DeepCASE.fit()` is equivalent to:
 1. :py:meth:`context_builder.ContextBuilder.fit()`
 2. :py:meth:`interpreter.Interpreter.fit()`

:py:meth:`module.DeepCASE.predict()` is equivalent to:
 1. :py:meth:`interpreter.Interpreter.predict()`

:py:meth:`module.DeepCASE.fit_predict()` is equivalent to:
 1. :py:meth:`module.DeepCASE.fit()`
 2. :py:meth:`module.DeepCASE.predict()`

Fit
~~~
The :py:meth:`fit()` method provides an API for directly learning the maliciousness score of sequences.
This method combines the ``fit()`` methods from both the :ref:`ContextBuilder` and the :ref:`Interpreter`.
We note that to use the :py:meth:`fit()` method, scores of sequences should be known a priori.
See the :py:meth:`interpreter.Interpreter.fit()` method for an explanation of how these scores are used.

.. automethod:: module.DeepCASE.fit

Predict
~~~~~~~
When DeepCASE is trained, we can use DeepCASE to predict the score of new sequences.
To this end, we provide the :py:meth:`predict()` function which takes ``context`` and ``events`` as input and outputs the labels of corresponding predicted clusters.
If no sequence could be matched, one of the following scores will be given:

 * ``-1``: Not confident enough for prediction
 * ``-2``: Label not in training
 * ``-3``: Closest cluster > epsilon

.. Note::

    This method is a wrapper around the :py:meth:`interpreter.Interpreter.predict()` method.

.. automethod:: module.DeepCASE.predict

Fit_predict
~~~~~~~~~~~
Similar to the ``scikit-learn`` API, the :py:meth:`fit_predict()` method performs the :py:meth:`fit()` and :py:meth:`predict()` functions in sequence on the same data.

.. automethod:: module.DeepCASE.fit_predict


I/O methods
^^^^^^^^^^^
DeepCASE can be saved and loaded from files using the following methods.
Please note that the :py:meth:`module.DeepCASE.load()` method is a ``classmethod`` and must be called statically.

.. automethod:: module.DeepCASE.save

.. automethod:: module.DeepCASE.load

**Example:**

.. code:: python

   from deepcase import DeepCASE
   deepcase = DeepCASE.load('<path_to_saved_deepcase_object>')
   deepcase.save('<path_to_save_deepcase_object>')
