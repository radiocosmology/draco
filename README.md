# CHIME Pipeline

This is the repository for storing the CHIME Pipeline.

Development should be done along the lines of [Github
Flow](https://guides.github.com/introduction/flow/). This is the same model we
use for `ch_util`.

Important notes:

 - *Don't* develop directly in master, use a feature branch for any change, and merge back into *master* promptly. Merging should be done by filing a Pull Request.
 - *Do* install the `virtualenv` with `./mkvenv.sh`

## Development Guidelines

The idea behind this repository is to keep track of the CHIME pipeline development, such that the union of the input data and this repository always gives the same output. This requires that we keep track of not only the code and scripts in this repository, but also any dependencies (discussed below).

As far as possible the pipeline code should be using Kiyo's pipeline task module `caput.pipeline`  ([doc](http://bao.phas.ubc.ca/codedoc/caput/)).

### Structure

Tasks should go into the appropriate subdirectory of `ch_pipeline/`. Ask for clarification if not clear.

### Coding Standards

Code should adhere to [PEP8](https://www.python.org/dev/peps/pep-0008/). If you
haven't looked at it please do so. We're a little more flexible on line length
than PEP8 (up to 100 characters is fine), but pretty strict on everything else.
Top tip is that decent editors (e.g. [Atom](http://atom.io/)) can be set to
automatically check PEP8 compliance as you code with tools like `Pylint`,
`flake8` or `autopep8`. Some of these also statically check your code for bugs,
which can be a real time saver.

Code should be well documented, with a docstring expected for each public
function, class or method. These should be done according to Numpy docstring
style
([guide](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)).

Code should also be well commented (not the same as well documented). Try and
put in comments that explain logically what each section of code is trying to
achieve, with specifics about any non-obvious parts.

### Branches

As mentioned above, this should be done a la Github Flow. Development should
generally be done around reasonably small, self contained, features, and should
be tracked in specific feature branches. These should not be long lived, as soon
as the feature is finished and tested, file a Pull Request, get it reviewed by
someone else, and when given the okay, merge the code and delete the branch. Any
new development should be branched off from `master` again.

Please don't use long lived, person specific branches e.g. `richard-dev`.

### Dependencies

Dependencies should be installable python packages. This means that they must
have a `setup.py` script in their root directory, and should be installable
using `python setup.py install`. They are kept track of in a `requirements.txt`
file. This file can contain references to exact versions of dependencies, using
both version tags, and commit hashes. An example `requirements.txt` file is
given below:
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

### Virtualenv

The script `mkvenv.sh` will automatically install a
[virtualenv](http://www.virtualenv.org/) containing all the pipeline
dependencies from the `requirements.txt` file. This gives a fresh, self-contained installation of the pipeline to work with. Before use, you should activate it using:
```bash
$ source venv/bin/activate
```
