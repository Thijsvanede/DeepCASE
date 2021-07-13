Installation
============
The most straightforward way of installing DeepCASE is via pip

.. note::

  Installation via pip is not yet possible.
  Please see installation `from source`_.

.. code::

  pip install deepcase

From source
^^^^^^^^^^^
If you wish to stay up to date with the latest development version, you can instead download the `source code`_.
In this case, make sure that you have all the required `dependencies`_ installed.
You can clone the code from GitHub:

.. code::

   git clone git@github.com:<REDACTED_FOR_ANONYMITY>/deepcase.git

Next, you can install the latest version using pip:

.. code::

  pip install -e <path/to/DeepCASE/directory/containing/setup.py>

.. _source code: https://github.com/<REDACTED_FOR_ANONYMITY>/DeepCASE

Dependencies
------------
DeepCASE requires the following python packages to be installed:

- Argformat: https://pypi.org/project/argformat/
- Matplotlib: https://matplotlib.org/
- NetworkX: https://networkx.github.io/
- Numpy: https://numpy.org
- Pandas: https://pandas.pydata.org/
- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/stable/index.html
- Scipy: https://www.scipy.org/
- Tqdm: https://tqdm.github.io/
- Urllib3: https://pypi.org/project/urllib3/

All dependencies should be automatically downloaded if you install FlowPrint via pip. However, should you want to install these libraries manually, you can install the dependencies using the requirements.txt file

.. code::

  pip install -r requirements.txt

Or you can install these libraries yourself

.. code::

  pip install -U argformat matplotlib networkx numpy pandas torch torchvision torchaudio scikit-learn scipy tqdm urllib3
