.. _DecoderAttention:

DecoderAttention
================
The DecoderAttention is an instance of the pytorch `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ class.
This part of the neural network takes the ``context_vector`` from the :ref:`Encoder` and produces the ``attention_vector``.

.. autoclass:: context_builder.decoders.DecoderAttention

.. automethod:: context_builder.decoders.DecoderAttention.__init__

Forward
^^^^^^^
The ``forward()`` function takes the ``context_vector`` and produces the ``attention_vector``.
This method is also called from the ``__call__`` method, i.e. when the object is called directly.

.. automethod:: context_builder.decoders.DecoderAttention.forward
