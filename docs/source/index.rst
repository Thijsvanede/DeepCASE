Welcome to DeepCASE's documentation!
====================================
This is the official documentation for the DeepCASE tool by the authors of the IEEE S&P `DeepCASE: Semi-Supervised Contextual Analysis of Security Events`_ paper.
Please cite this work when using the software for academic research papers, see :ref:`Citing` for more information.

.. _`DeepCASE: Semi-Supervised Contextual Analysis of Security Events`: https://vm-thijs.ewi.utwente.nl/static/homepage/papers/deepcase.pdf

DeepCASE introduces a semi-supervised approach for the contextual analysis of security events.
This approach automatically finds correlations in sequences of security events and clusters these correlated sequences.
The clusters of correlated sequences are then shown to security operators who can set policies for each sequence.
Such policies can ignore sequences of unimportant events, pass sequences to a human operator for further inspection, or (in the future) automatically trigger response mechanisms.
The main contribution of this work is to reduce the number of manual inspection security operators have to perform on the vast amounts of security events that they receive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage/usage
   reference/reference
   roadmap
   contributors
   license
   citing
