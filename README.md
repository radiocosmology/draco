# CHIME Pipeline

This is the repository for storing the CHIME Pipeline. 

Important notes:

 - *Don't* develop in master, use a `passX` branch instead (`git checkout passX`)
 - *Do* install the `virtualenv` with `./mkvenv.sh`

## Development Guidelines

The idea behind this repository is to keep track of the CHIME pipeline development, such that the union of the input data and this repository always gives the same output. This requires that we keep track of not only the code and scripts in this repository, but also any dependencies (discussed below).

### Branches

Development should be done in `passX` branches, or in feature branches that are
merged back in.

### Dependencies

Dependencies should be installable python packages. They are kept track of in a
`requirements.txt` file. This file can contain references to exact versions of
dependencies, using both version tags, and commit hashes. An example `requirements.txt` file is given below:

```
-e git+https://github.com/radiocosmology/caput@ee1c55ea4cf8cb7857af2ef3adcb2439d876768d#egg=caput-master
-e git+ssh://git@bitbucket.org/chime/ch_util.git@e33b174696509b158c15cf0bfc27f4cb2b0c6406#egg=ch_util-e
-e git+https://github.com/radiocosmology/cora@v1.0.0#egg=cora
-e git+https://github.com/radiocosmology/driftscan@v1.0.0#egg=driftscan

```
Here, the first two requirements specify an exact git hash, whereas the second two use git tags as a shorthand.

These dependencies can be installed using:
```bash
$ pip install -r requirements.txt
```
This is automatically done by the `mkvenv.sh` script.

### Structure