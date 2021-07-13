.. _VarAdam:

VarAdam
=======
The VarAdam is an instance of the pytorch `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ class.
The VarAdam class implements an adapted version of the Adam optimizer as introduced in [1].

.. autoclass:: context_builder.optimizer.VarAdam

.. automethod:: context_builder.optimizer.VarAdam.__init__

Update
^^^^^^
The following functions update the optimizer with a given number of steps.

.. automethod:: context_builder.optimizer.VarAdam.step

.. automethod:: context_builder.optimizer.VarAdam.rate

Reference
^^^^^^^^^
 [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In Advances in neural information processing systems (NIPS). [`PDF`_]

 .. _PDF: https://dl.acm.org/doi/10.5555/3295222.3295349
