.. _Preprocessor:

Preprocessor
============
The Preprocessor class provides methods to automatically extract DeepCASE event sequences from various common data formats.
To start sequencing, first create the Preprocessor object.

.. autoclass:: preprocessing.Preprocessor

.. automethod:: preprocessing.Preprocessor.__init__

Sequencing
^^^^^^^^^^
All supported formats are wrappers around the sequence method which will produce context and event sequences from given events.

.. automethod:: preprocessing.Preprocessor.sequence

Formats
^^^^^^^
We currently support the following formats:
 * ``.csv`` files containing a header row that specifies the columns 'timestamp', 'event' and 'machine'.
 * ``.txt`` files containing a line for each machine and a sequence of events (integers) separated by spaces.

Transforming ``.csv`` files into DeepCASE sequences is the quickest method and is done by the following method call:

.. automethod:: preprocessing.Preprocessor.csv

Transforming ``.txt`` files into DeepCASE sequences is slower, but still possible using the following method call:

.. automethod:: preprocessing.Preprocessor.text

Future supported formats
------------------------

.. note::

   These formats already have an API entrance, but are currently **NOT** supported.

* ``.json`` files containing values for 'timestamp', 'event' and 'machine'.
* ``.ndjson`` where each line contains a json file with keys 'timestamp', 'event' and 'machine'.

.. automethod:: preprocessing.Preprocessor.json

.. automethod:: preprocessing.Preprocessor.ndjson
