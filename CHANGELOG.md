# Change Log

All notable changes to this project will be documented in this file. This
project adheres to [Semantic Versioning](http://semver.org/), with the exception
that I'm using PEP440 to denote pre-releases.

## [0.2.0] - 2017-06-24

### Added

- MPI aware logging functionality for tasks
- Support for table like data containers
- Routines for delay spectrum estimation and filtering
- KL based foreground filtering
- SVD based foreground filtering
- Quadratic power spectrum estimation
- Task for setting weights based on the radiometer expectation
- Baseline masking task
- Support for reading `ProductManager` instances (from `driftscan`) and updated
  tasks to allow using them in place of `BeamTransfer` or `TransitTelescope`
  instances as task arguments.

### Fixes

- Added a missing routine `_ensure_list`.


## [0.1.0] - 2016-08-11

### Added

- Initial version of `draco` importing code from the CHIME pipeline.
