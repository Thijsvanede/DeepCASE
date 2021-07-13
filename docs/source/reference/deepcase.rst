.. _DeepCASE:

DeepCASE
========
The DeepCASE class is a wrapper class that combines the :ref:`ContextBuilder` and :ref:`Interpreter`.

.. py:class:: DeepCASE(*args: Any, **kwargs, Any)

.. py:method:: DeepCASE.__init__(input_size, output_size, hidden_size=128, num_layers=1, max_length=10, bidirectional=False, LSTM=False)

    Analyse security events with respect to contextual machine behaviour

    :param int n_features: Number of different possible security events

    :param int context: Maximum size of context window as number of events
    :type context: int, default=10

    :param int complexity: Complexity of the network, describes the number of hidden dimensions used by the sequence prediction algorithm.
    :type complexity: int, default=128

    :param string device: Hardware device to use for computation. If None, fastest hardware is automatically inferred. If 'cuda' a specific graphics card can be selected using cuda<i>, where <i> is the identifier of the graphics card, e.g. ``"cuda0"`` for card 0.
    :type device: string, (cpu|cuda<i?>|None), default=None

    :param float eps: Epsilon used for determining maximum distance between clusters.
    :type eps: float, default=0.1

    :param int min_samples: Minimum number of required samples per cluster.
    :type min_samples: int, default=5

    :param float threshold: Minimum required confidence in fingerprint before using it in training clusters.
    :type threshold: float, default=0.2

Accessing components
^^^^^^^^^^^^^^^^^^^^
The :ref:`ContextBuilder` and :ref:`Interpreter` can be accessed through the following attributes:

.. py:attribute:: DeepCASE.context_builder

    Attribute holding the :ref:`ContextBuilder` object.

.. py:attribute:: DeepCASE.interpreter

Attribute holding the :ref:`Interpreter` object.
