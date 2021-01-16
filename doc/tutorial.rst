Tutorial
--------

This tutorial is going to go through the process of generating BAO constraints
from the Pathfinder data. Just kidding! We're actually just going to generate
some simulated data and the turn it into maps.

Setting up the Pipeline
^^^^^^^^^^^^^^^^^^^^^^^

Before you start, make sure you have access to the CHIME bitbucket organisation,
and have set up your ssh keys for access to the bitbucket repositories from the
machine you want to run the pipeline on. Unless you are working on `Scinet`,
you'll also want to ensure you have an account on `niedermayer` and your ssh
keys are set up to allow password-less login, to ensure the database connection
can be set up.

There are a few software pre-requesites to ensure you have installed. Obviously
python is one of them, with `numpy` and `scipy` installed, but you also need to
have `virtualenv`, allowing us to install the pipeline and it's dependencies
without messing up the base python installation. To check you have it installed
try running::

    $ virtualenv --help

if you get an error, it's not installed properly so you'll need to fix it.

With that all sorted, we're ready to start. First step, download the pipeline
repository to wherever you want it installed::

    $ git clone git@github.com/radiocosmology/draco.git

Then change into that directory, and run the script `mkvenv.sh`::

    $ cd draco
    $ ./mkvenv.sh

The script will do three things. First it will create a python virtual
environment to isolate the CHIME pipeline installation. Second it will fetch the
python pre-requesites for the pipeline and install them into the virtualenv.
Finally, it will install itself into the new virtualenv. Look carefully through
the messages output for errors to make sure it completed successfully. You'll
need to activate the environment whenever you want to use the pipeline. To do
that, simply do::

    $ source <path to pipeline>/venv/bin/activate

You can check that it's installed correctly by firing up python, and attempting
to import some of the packages. For example::

    >>> from drift.core import telescope
    >>> print telescope.__file__
    /Users/richard/code/draco/venv/src/driftscan/drift/core/telescope.pyc
    >>> from draco import containers
    >>> print containers.__file__
    /Users/richard/code/draco/draco/containers.pyc


External Products
^^^^^^^^^^^^^^^^^

If you are here, you've got the pipeline successfully installed. Congratulations.

There are a few data products we'll need to run the pipeline that must be
generated externally. Fortunately installing the pipeline has already setup all
the tools we need to do this.

We'll start with the beam transfer matrices, which describes how the sky gets
mapped into our measured visibilities. These are used both for simulating
observations given a sky map, and for making maps from visibilities (real or
simulated). To generate them we use the `driftscan` package, telling it what
exactly to generate with a `YAML` configuration file such as the one below.

.. literalinclude:: product_params.yaml
    :linenos:
    :language: YAML

This file is run with the command::

    $ drift-makeproducts run product_params.yaml

To simulate the timestreams we also need a sky map to base it on. The ``cora``
package contains several different sky models we can use to produce a sky map.
The easiest method is to use the `cora-makesky` command, e.g.::

    $ cora-makesky foreground 64 401.0 411.0 5 foreground_map.h5

which will generate an ``HDF5`` file containing simulated foreground maps at each
polarisation (Stokes I, Q, U and V) with five frequency channels between 401.0
and 411.0 MHz. Each map is in Healpix format with ``NSIDE=16``. There are options
to produce 21cm signal simulations as well as point source only, and galactic
synchrotron maps.

Map-making with the Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CHIME pipeline is built using the infrastructure developed by Kiyo in the
``caput.pipeline`` module. Python classes are written to perform task on the
data, and a YAML configuration file describes how these should be configured and
connected together. Below I've put the configuration file we are going to use to
make maps from simulated data:

.. literalinclude:: pipeline_params.yaml
    :linenos:
    :language: YAML

Before we jump into making the maps, let's briefly go over what this all means.
For further details you can consult the ``caput`` documentation on the pipeline.

The bulk of this configuration file is a list of tasks being configured. There
is a ``type`` field where the class is specified by its fully qualified python
name (for example, the first task ``draco.io.LoadBeamTransfer``). To
connect one task to another, you simply specify a label for the ``output`` of
one task, and give the same label to the ``input`` or ``requires`` of the other
task. The labels themselves are dummy variables, any string will do, provided it
does not clash with the name of another label. The distinction between ``input``
and ``requires`` is that the first is for an input which is passed every cycle
of the pipeline, and the second is for something required only at initialisation
of the task.

Often we might want to configure a task from the YAML file itself. This is done
with the ``params`` section of each task. The named items within this section
are passed to the pipeline class when it is created. Each entry corresponds to a
``config.Property`` attribute on the class. For example the ``SimulateSidereal``
class has parameters that can be specified::

    class SimulateSidereal(task.SingleTask):
        """Create a simulated timestream.

        Attributes
        ----------
        maps : list
            List of map filenames. The sum of these form the simulated sky.
        ndays : float, optional
            Number of days of observation. Setting `ndays = None` (default) uses
            the default stored in the telescope object; `ndays = 0`, assumes the
            observation time is infinite so that the noise is zero. This allows a
            fractional number to account for higher noise.
        seed : integer, optional
            Set the random seed used for the noise simulations. Default (None) is
            to choose a random seed.
        """
        maps = config.Property(proptype=list)
        ndays = config.Property(proptype=float, default=0.0)
        seed = config.Property(proptype=int, default=None)

        ...

In the YAML file we configured the task as follows:

.. code-block:: YAML

    -   type:       draco.synthesis.stream.SimulateSidereal
        requires:   [tel, bt]
        out:        sstream
        params:
            maps:   [ "testfg.h5" ]
            save:   Yes
            output_root: teststream_

Of the three properties available from the definition of ``SimulateSidereal`` we
have only configured one of them, the list of maps to process. The remaining two
entries of the ``params`` section are inherited from the pipeline base task.
These simply tell the pipeline to save the output of the task, with a base name
given by ``output_root``.

The pipeline is run with the `caput-pipeline` script::

    $ caput-pipeline run pipeline_params.yaml

What has it actually done? Let's just quickly go through the tasks in order:

#. Load the beam transfer manager from disk. This just gives the pipeline access
   to all the beam transfer matrices produced by the driftscan code.

#. Load a map from disk, use the beam transfers to transform it into a sidereal timestream.

#. Select the products from the timestream that are understood by the given beam
   transfer manager. In this case it won't change anything, but this task can
   subset frequencies and products as well as average over redundant baselines.

#. Perform the m-mode transform on the sidereal timestream.

#. Apply the map maker to the m-modes to produce a dirty map.

#. Apply the map maker to the generate a Wiener filtered map.

Ninja Techniques
^^^^^^^^^^^^^^^^

Running on a cluster. Coming soon....
