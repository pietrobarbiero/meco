MECO: Multi-objective Evolutionary Compression
======================================================

|Build|
|Coverage|
|Dependendencies|
|PyPI license|
|PyPI-version|



.. |Build| image:: https://img.shields.io/travis/pietrobarbiero/meco?label=Master%20Build&style=for-the-badge
    :alt: Travis (.org)
    :target: https://travis-ci.com/pietrobarbiero/meco

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/pietrobarbiero/meco?label=Test%20Coverage&style=for-the-badge
    :alt: Codecov
    :target: https://codecov.io/gh/pietrobarbiero/meco

.. |Dependendencies| image:: https://img.shields.io/requires/github/pietrobarbiero/meco?style=for-the-badge
    :alt: Requires.io
    :target: https://requires.io/github/pietrobarbiero/meco/requirements/?branch=master

.. |PyPI license| image:: https://img.shields.io/pypi/l/meco.svg?style=for-the-badge
   :target: https://pypi.python.org/pypi/meco/

.. |PyPI-version| image:: https://img.shields.io/pypi/v/meco?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.python.org/pypi/meco/

The MECO (Multi-objective Evolutionary
COmpression) algorithm is a tool to perform:

* dataset compression,
* feature selection, and
* coreset discovery.


This python package provides a sklearn-like transformer
implementation of the MECO algorithm.

Quick start
-----------

You can install the ``meco`` package along with all its dependencies from
`PyPI <https://pypi.org/project/meco/>`__:

.. code:: bash

    $ pip install meco


Example
------------

For this simple experiment, let's use the `digits <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html>`__
dataset from sklearn. We first need to import the dataset,
a simple sklearn classifier (e.g. Ridge) and the ``MECO`` transformer.
We can then load the dataset, create a ``MECO`` model, and
fit the model on the ``digits`` dataset:

.. code:: python

    from sklearn.datasets import load_digits
    from sklearn.linear_model import RidgeClassifier

    from meco import MECO

    X, y = load_digits(return_X_y=True)

    model = MECO(RidgeClassifier(random_state=42))
    model.fit(X, y)

Once training is over, we get a view of the `compressed`
input data ``X`` containing the most relevant samples
(i.e. a subset of the rows in ``X``, a.k.a. the `coreset`),
and the most relevant features (i.e. a subset of the columns in ``X``):

.. code:: python

    x_reduced = model.transform(X)

Once trained, the ``model.best_set_`` dictionary contains:

* the indices of the most relevant samples,
* the indices of the most relevant features, and
* the validation accuracy of the compressed dataset ``x_reduced``, e.g.:

.. code:: python

    >>> model.best_set_
    {
        'samples': [0, 2, 4, ...],
        'features': [3, 7, 8, ...],
        'accuracy': 0.9219,
    }

The compressed dataset ``(x_reduced, y_reduced)`` can be used
instead of the original dataset ``(X, y)`` to train machine
learning models more efficiently:

.. code:: python

    from sklearn.ensemble import RandomForestClassifier

    y_reduced = y[model.best_set_['samples']]

    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(x_reduced, y_reduced)

Tasks
-----

Dataset compression
^^^^^^^^^^^^^^^^^^^^^^^^^

Should you need to compress the **whole** dataset ``X``
(i.e. for dataset compression), you can set the parameter ``compression``
to ``'both'`` (this is the **default** behaviour anyway):

.. code:: python

    model = MECO(RidgeClassifier(), compression='both')


Coreset discovery
^^^^^^^^^^^^^^^^^^^^^^^^^

Should you need to compress the **rows** of ``X`` only
(i.e. for coreset discovery), you can set the parameter ``compression``
to ``'samples'``:

.. code:: python

    model = MECO(RidgeClassifier(), compression='samples')


Feature selection
^^^^^^^^^^^^^^^^^^^^^^^^^

Should you need to compress the **columns** of ``X`` only
(i.e. for feature selection), you can set the parameter ``compression``
to ``'features'``:

.. code:: python

    model = MECO(RidgeClassifier(), compression='features')



Citing
----------

If you find MECO useful in your research, please consider citing the following papers::

    @inproceedings{barbiero2019novel,
      title={A Novel Outlook on Feature Selection as a Multi-objective Problem},
      author={Barbiero, Pietro and Lutton, Evelyne and Squillero, Giovanni and Tonda, Alberto},
      booktitle={International Conference on Artificial Evolution (Evolution Artificielle)},
      pages={68--81},
      year={2019},
      organization={Springer}
    }

    @article{barbiero2020uncovering,
      title={Uncovering Coresets for Classification With Multi-Objective Evolutionary Algorithms},
      author={Barbiero, Pietro and Squillero, Giovanni and Tonda, Alberto},
      journal={arXiv preprint arXiv:2002.08645},
      year={2020}
    }


Source
------

The source code and minimal working examples can be found on
`GitHub <https://github.com/pietrobarbiero/meco>`__.


Authors
-------

`Pietro Barbiero <http://www.pietrobarbiero.eu/>`__,
`Giovanni Squillero <https://staff.polito.it/giovanni.squillero/>`__,
and
`Alberto Tonda <https://www.researchgate.net/profile/Alberto_Tonda>`__.

Licence
-------

Copyright 2020 Pietro Barbiero, Giovanni Squillero, and Alberto Tonda.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.