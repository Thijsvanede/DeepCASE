.. _DecoderEvent:

DecoderEvent
============
The DecoderEvent is an instance of the pytorch `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ class.
This part of the neural network takes the encoded inputs from the :ref:`Encoder` and ``attention_vector`` from the :ref:`DecoderAttention` and predicts the next event in the sequence.

.. autoclass:: context_builder.decoders.DecoderEvent

.. automethod:: context_builder.decoders.DecoderEvent.__init__

Forward
^^^^^^^
The ``forward()`` function takes the ``attention_vector`` and encoded inputs and predicts the next event in the sequence.
This method is also called from the ``__call__`` method, i.e. when the object is called directly.

.. automethod:: context_builder.decoders.DecoderEvent.forward
