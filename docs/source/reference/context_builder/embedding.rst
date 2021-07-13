.. _EmbeddingOneHot:

EmbeddingOneHot
===============
The EmbeddingOneHot is an instance of the pytorch `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ class.
This part of the neural network takes categorical samples and produces a one-hot encoded version of the input.
This module is used in the from the :ref:`Encoder`.

.. autoclass:: context_builder.embedding.EmbeddingOneHot

.. automethod:: context_builder.embedding.EmbeddingOneHot.__init__

Forward
^^^^^^^
The ``forward()`` function takes the input values and produces the one-hot encoded equivalent.
This method is also called from the ``__call__`` method, i.e. when the object is called directly.

.. automethod:: context_builder.embedding.EmbeddingOneHot.forward
