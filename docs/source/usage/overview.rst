Overview
========
This section gives a high-level overview of the different steps taken by DeepCASE to perform a contextual analysis of security events and explains how DeepCASE clusters events to reduce the workload of security analysts.

1) `Event sequencing`_

2) `Context Builder`_

3) `Interpreter`_

4) `Manual Analysis`_

5) `Semi-automatic Analysis`_

.. figure:: ../_static/overview.png

    Figure 1: Overview of DeepCASE.

Event sequencing
^^^^^^^^^^^^^^^^
The first step is to transform events stored in your local format into a format that DeepCASE can handle.
For this step, we use the :ref:`Preprocessor` class, which is able to take events stored in a ``.csv`` and ``.txt`` format and transform them into DeepCASE sequences.
For the required formats for both the ``.csv`` and ``.txt`` files, we refer to the :ref:`Preprocessor` reference.

Context Builder
^^^^^^^^^^^^^^^
Next, DeepCASE passes the sequences to the  :ref:`ContextBuilder`.
When receiving sequences, the :ref:`ContextBuilder` first applies its :py:meth:`fit()` method to train its neural network.
Once the network is trained, we use the :ref:`ContextBuilder`'s  :py:meth:`predict()` method to get the ``confidence`` in each event with its context and ``attention`` for all events in the context.
These ``confidence`` and ``attention`` values can then be passed to the :ref:`Interpreter` together with the ``events`` and their ``context`` for clustering.

.. figure:: ../_static/architecture.png

    Figure 2: Architecture of DeepCASE's Context Builder.

Interpreter
^^^^^^^^^^^
The :ref:`Interpreter` takes the ``events``, ``context``, ``confidence``, and ``attention`` to cluster similar event contexts.
To this end, the :ref:`Interpreter` first performs an `attention query` using the :ref:`ContextBuilder`'s  :py:meth:`query()` method.
Once this query is performed, it uses the ``attention`` values to weigh each ``context`` event.
This weighed ``context`` is then clustered according to their L1 similarity.

The :ref:`Interpreter` provides a :py:meth:`fit()` method to create the clusters and a :py:meth:`predict()` method to match against known clusters.
These methods will be used for the Manual Analysis and Semi-automatic Analysis.

Manual Analysis
^^^^^^^^^^^^^^^
TODO

Semi-automatic Analysis
^^^^^^^^^^^^^^^^^^^^^^^
TODO
