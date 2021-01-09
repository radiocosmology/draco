# draco

A pipeline for the analysis and simulation of drift scan radio data.

**draco** is a set of building blocks designed for the analysis of the transit
radio data with the m-mode formalism (see the papers
[arXiv:1302.0327](http://arxiv.org/abs/1302.0327) and
[arXiv:1401.2095](http://arxiv.org/abs/1401.2095) for details). It is being used
as part of the analysis and simulation pipeline for
[CHIME](http://chime.phas.ubc.ca) though is (and will remain) telescope agnostic. It
can:

- Simulate time stream data from maps of the sky (using the m-mode formalism)
- Add gain fluctuations and correctly correlated instrumental noise (i.e.
  Wishart distributed)
- Perform various cuts on the data
- Make maps of the sky from data using the m-mode formalism

It does not do some of the key steps in radio data analysis, notably *RFI
flagging* and *calibration*. The implementations we had were too specific to
CHIME so they have been left out, until a more generic version is produced.

To do this it depends on various related packages:

- [driftscan](http://github.com/radiocosmology/driftscan): for modelling the
  telescope and generating the computationally intensive products required for
  simulation and analysis.
- [cora](http://github.com/radiocosmology/cora): for modelling and simulating
  the radio sky
- [caput](http://github.com/radiocosmology/caput): provides infrastructure for
  building these packages

It also depends on the usual suspects: `numpy`, `scipy`, `healpy`, `h5py` and `skyfield`.

draco can be installed with `pip` in the usual way:
```
$ pip install git+https://github.com/radiocosmology/draco.git
```
or by downloading the package and running the `setup.py` script:
```
$ cd draco
$ python setup.py install
```

The full documentation of `draco` is at https://radiocosmology.github.io/draco/.
