.. _Encoder:

Encoder
=======
The Encoder is an instance of the pytorch `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ class.
This part of the neural network takes the input sequences and produces the embedded outputs as well as the ``context_vector`` used by the :ref:`DecoderAttention` and :ref:`DecoderEvent`.

.. autoclass:: context_builder.encoders.Encoder

.. automethod:: context_builder.encoders.Encoder.__init__

Forward
^^^^^^^
The ``forward()`` function takes the input sequences and produces the embedded outputs as well as the ``context_vector``.
This method is also called from the ``__call__`` method, i.e. when the object is called directly.

.. automethod:: context_builder.encoders.Encoder.forward
