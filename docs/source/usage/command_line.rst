Command line tool
=================
When DeepCASE is installed, it can be used from the command line.
The ``__main__.py`` file in the ``deepcase`` module implements this command line tool.
The command line tool provides a quick and easy interface to predict sequences from ``.csv`` or ``.txt`` files.
The full command line usage is given in its help page:

.. code::

   usage: deepcase.py [-h] [--csv CSV] [--txt TXT] [--events EVENTS] [--length LENGTH] [--timeout TIMEOUT]
                      [--save-sequences SAVE_SEQUENCES] [--load-sequences LOAD_SEQUENCES] [--hidden HIDDEN]
                      [--delta DELTA] [--save-builder SAVE_BUILDER] [--load-builder LOAD_BUILDER]
                      [--confidence CONFIDENCE] [--epsilon EPSILON] [--min_samples MIN_SAMPLES]
                      [--save-interpreter SAVE_INTERPRETER] [--load-interpreter LOAD_INTERPRETER]
                      [--save-clusters SAVE_CLUSTERS] [--load-clusters LOAD_CLUSTERS]
                      [--save-prediction SAVE_PREDICTION] [--epochs EPOCHS] [--batch BATCH] [--device DEVICE]
                      [--silent]
                      {sequence,train,cluster,manual,automatic}

   DeepCASE: Semi-Supervised Contextual Analysis of Security Events

   positional arguments:
     {sequence,train,cluster,manual,automatic}  mode in which to run DeepCASE

   optional arguments:
     -h, --help                                 show this help message and exit

   Input/Output:
     --csv CSV                                  CSV events file to process
     --txt TXT                                  TXT events file to process
     --events EVENTS                            number of distinct events to handle         (default =  auto)

   Sequencing:
     --length LENGTH                            sequence LENGTH                             (default =    10)
     --timeout TIMEOUT                          sequence TIMEOUT (seconds)                  (default = 86400)
     --save-sequences SAVE_SEQUENCES            path to save sequences
     --load-sequences LOAD_SEQUENCES            path to load sequences

   ContextBuilder:
     --hidden HIDDEN                            HIDDEN layers dimension                     (default =   128)
     --delta DELTA                              label smoothing DELTA                       (default =   0.1)
     --save-builder SAVE_BUILDER                path to save ContextBuilder
     --load-builder LOAD_BUILDER                path to load ContextBuilder

   Interpreter:
     --confidence CONFIDENCE                    minimum required CONFIDENCE                 (default =   0.2)
     --epsilon EPSILON                          DBSCAN clustering EPSILON                   (default =   0.1)
     --min_samples MIN_SAMPLES                  DBSCAN clustering MIN_SAMPLES               (default =     5)
     --save-interpreter SAVE_INTERPRETER        path to save Interpreter
     --load-interpreter LOAD_INTERPRETER        path to load Interpreter
     --save-clusters SAVE_CLUSTERS              path to CSV file to save clusters
     --load-clusters LOAD_CLUSTERS              path to CSV file to load clusters
     --save-prediction SAVE_PREDICTION          path to CSV file to save prediction

   Train:
     --epochs EPOCHS                            number of epochs to train with              (default =    10)
     --batch BATCH                              batch size       to train with              (default =   128)

   Other:
     --device DEVICE                            DEVICE used for computation (cpu|cuda|auto) (default =  auto)
     --silent                                   silence mode, do not print progress

Examples
^^^^^^^^
Below, we provide various examples of using the command-line tool for running DeepCASE.

Event sequencing
~~~~~~~~~~~~~~~~
Transform ``.csv`` or ``.txt`` files into sequences and store them in the file ``sequences.save``.

.. code::

   python3 deepcase sequence --csv <path/to/file.csv> --save-sequences sequences.save
   python3 deepcase sequence --txt <path/to/file.txt> --save-sequences sequences.save

ContextBuilder
~~~~~~~~~~~~~~
Train the :ref:`ContextBuilder` on the input samples loaded from the file ``sequences.save`` and store the trained ContextBuilder in the file ``builder.save``.

.. code::

   python3 deepcase train\
        --load-sequences sequences.save\
        --save-builder builder.save

Interpreter
~~~~~~~~~~~
Run in manual mode where the :ref:`Interpreter` clusters the given sequences.
We load the sequences from ``sequences.save`` and the trained ContextBuilder from ``builder.save``.
We store the interpreter (containing all clusters) to the file ``interpreter.save`` and the generated clusters to ``clusters.csv``.
The ``clusters.csv`` file contains two columns: ``cluster`` and ``label``.
We can manually label the individual samples within the cluster by changing the ``label`` value, note that the rows of the csv file corresond to the loaded sequences.
If the sequences itself contained labels, these labels are used for storing in the ``csv`` file, otherwise, all clusters are assigned a label of ``-1``.

.. code::

   python3 deepcase cluster\
        --load-sequences sequences.save\
        --load-builder builder.save\
        --save-interpreter interpreter.save\
        --save-clusters clusters.csv

Manual Mode
~~~~~~~~~~~
Once we (manually) provided a label to each cluster, we can assign these label in manual mode and save the updated interpreter.

.. note::

   If ``--load-clusters`` is not specified, DeepCASE will try to use the labels extracted from the ``sequences`` it processes (see :ref:`Preprocessor`).
   If no ``labels`` were provided there either, DeepCASE throws an error.

.. code::

   python3 deepcase manual\
        --load-sequences sequences.save\
        --load-builder builder.save\
        --load-interpreter interpreter.save\
        --load-clusters clusters.csv\
        --save-interpreter interpreter_fitted.save

(Semi)-automatic Mode
~~~~~~~~~~~~~~~~~~~~~
Once we assigned labels to the clusters in the :ref:`Interpreter`, we can use DeepCASE to predict labels for new sequences.
We save these predicted labels in a file called ``prediction.save``.

.. note::

    If sequences contain ``labels`` (see :ref:`Preprocessor`), we also output a classification report and confusion matrix to show the performance of DeepCASE.

.. code::

   python3 deepcase automatic\
        --load-sequences sequences.save\
        --load-builder builder.save\
        --load-interpreter interpreter_fitted.save\
        --save-prediction prediction.csv
