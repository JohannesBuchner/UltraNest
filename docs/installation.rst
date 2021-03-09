.. _install:
.. highlight:: shell

============
Installation
============


Stable release
--------------

To install UltraNest, run this command in your terminal:

.. code-block:: console

    $ pip install ultranest

This is the preferred method to install UltraNest, as it will always install the most recent stable release.

If you get this error:

    ModuleNotFoundError: No module named 'Cython'

run:

.. code-block:: console

    $ pip install cython
    $ pip install ultranest

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

If you use conda, you can install UltraNest with:

.. code-block:: console

    $ conda install --channel conda-forge ultranest

From sources
------------

The sources for UltraNest can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/JohannesBuchner/ultranest

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/JohannesBuchner/ultranest/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/JohannesBuchner/ultranest
.. _tarball: https://github.com/JohannesBuchner/ultranest/tarball/master
