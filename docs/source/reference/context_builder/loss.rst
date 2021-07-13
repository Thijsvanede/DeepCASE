.. _LabelSmoothing:

LabelSmoothing
==============
The LabelSmoothing is an instance of the pytorch `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ class.
The LabelSmoothing class implements an adapted version of the label smoothing loss function [1].

.. autoclass:: context_builder.loss.LabelSmoothing

.. automethod:: context_builder.loss.LabelSmoothing.__init__

Forward
^^^^^^^
The ``forward()`` function takes actual output ``x`` and ``target`` output and computes the loss.
This method is also called from the ``__call__`` method, i.e. when the object is called directly.

.. automethod:: context_builder.loss.LabelSmoothing.forward

Reference
^^^^^^^^^
 [1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the Inception Architecture for Computer Vision. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR). [`PDF`_]

 .. _PDF: https://doi.org/10.1109/CVPR.2016.308
