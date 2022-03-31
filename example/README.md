# Examples
Here we list three examples of how to use DeepCASE.

## Regular usage
As explained in the [documentation](deepcase.readthedocs.io), DeepCASE offers two interfaces:
 1. The `DeepCASE` approach as described in the paper, with each individual step as a separate method. This includes methods where clusters can be manually labelled in a separate step. This is implemented in `example.py`
 2. A `DeepCASE` module that can be used to `fit` and `predict` samples using a single `fit_predict` method. Note that this method only works if we already have labelled the sequences we input into DeepCASE. We show an example of using this interface in `example_module.py`.

## Context Builder Sequence Prediction
Besides using the entire workflow, we can also use DeepCASE's ContextBuilder to predict the next item in a sequence.
`example_hdfs.py` gives an example on how to use the DeepCASE's ContextBuilder to predict the next item in the HDFS dataset (Table IV in paper).
