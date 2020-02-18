# [20.2.0](https://github.com/radiocosmology/draco/compare/v0.2.0...v20.2.0) (2020-02-18)

This release switched to calendar versioning.

### Bug Fixes

* **apply_gains:** fixes deprecation issue with non tuple slices ([c915d6e](https://github.com/radiocosmology/draco/commit/c915d6e388db7dba8e7540f5d4ddbfa2c6d1e086))
* **containers:** fix for hangs/crashes when writing HDF5 containers ([1e399a5](https://github.com/radiocosmology/draco/commit/1e399a523b439f69f6f3d2fe28306e5d3546a499))
* **containers,io,task:** compare to `basestring` from `past.builtins`. ([066315d](https://github.com/radiocosmology/draco/commit/066315d3a8d46c726b015c9a61dbbc8dcf579191))
* **DelayFilter:** updates to worked with new stacked data ([d78f739](https://github.com/radiocosmology/draco/commit/d78f739d57905b6cf8ee35eafee8b1f571478b4d))
* **flagging:** corrected calculation of TV channel flagging threshold ([b398f1d](https://github.com/radiocosmology/draco/commit/b398f1df7c13fef55a1103599514ba4252446c77))
* **gains.py:** removed unnecessary private package dependency ([8c78b27](https://github.com/radiocosmology/draco/commit/8c78b2715b40099f1cc791a83100c627afb59525))
* **GaussianNoise:** fix issue when adding noise to stacked data ([e30d018](https://github.com/radiocosmology/draco/commit/e30d0186de73ed256ce75620c691f9556752b184))
* **LoadMaps:** fix issue in Python 3 porting that stopped maps loading ([f4859a8](https://github.com/radiocosmology/draco/commit/f4859a8e45149a61984a52310fbffeb81a53eaf2))
* **MaskData:** loop over prodstack, global slicing of weights outside of loop ([4278f56](https://github.com/radiocosmology/draco/commit/4278f5602a953c17ff5b68fbd678788b954a61c6))
* **misc:** expand gain weights to correct shape ([df71e22](https://github.com/radiocosmology/draco/commit/df71e22fc14788b8635b92d093d7e4f8bbf60cb5))
* **RandomGains:** fixes regressions introduced in [#20](https://github.com/radiocosmology/draco/issues/20) ([e38cdc1](https://github.com/radiocosmology/draco/commit/e38cdc194c5c5aa6cec20f13951ac6fe6006a715))
* **ReceiverTemperature:** solve crash when applying to stacked data ([b2181dd](https://github.com/radiocosmology/draco/commit/b2181dd3c149453d2d33a93e0ee45379851040b9))
* **RFIMask:** fixes for broken RFIMask task ([52a7c31](https://github.com/radiocosmology/draco/commit/52a7c31426e2968bc4c60c76e0337f3ec7098720))
* **setup.py:** removed unicode_literals for setuptools support ([f8e139f](https://github.com/radiocosmology/draco/commit/f8e139f9da66270770c334776c0e1ede10fafb08))
* **SiderealGrouper:** respect the padding option ([a8c9a6f](https://github.com/radiocosmology/draco/commit/a8c9a6f7820437b55372889db1b48783212fa00d))
* **SimulateSidereal:** issue checking frequencies between map and telescope ([eaa0376](https://github.com/radiocosmology/draco/commit/eaa0376fbbdbe06772cbcba0fd6bbd8a28145fb5))
* **task:** log what is being deleted. ([838530b](https://github.com/radiocosmology/draco/commit/838530b3ca7fb74835d7253dbcf0580f5ad4f6dc))
* **transform:** change how `bt_rev` is calculated. ([b3c5a0f](https://github.com/radiocosmology/draco/commit/b3c5a0f1813561c227b3460044661f9db9f9a38f))
* **transform:** python 3 fix for find_key ([06b0d7b](https://github.com/radiocosmology/draco/commit/06b0d7bd7bbbb8666b53c0a2ed73134dd8419983))
* **transform:** try to redefine stack axis using only unmasked products ([dde6b6c](https://github.com/radiocosmology/draco/commit/dde6b6c128eca3adf41b04eef65404550b37b0c4))
* **transform:** use `time` attribute instead of `index_map['time']` ([977d5bd](https://github.com/radiocosmology/draco/commit/977d5bdb620c0d810bb52f0e08a01b6f44d7f57b))
* **transform._pack_marray:** broadcast mmodes to correct shape before assignment to marray ([f527356](https://github.com/radiocosmology/draco/commit/f52735643379917629dbdcfa72b6aa0be2798268))
* **transform._pack_marray:** Missing highest modes when packing FFT in ([1d4730e](https://github.com/radiocosmology/draco/commit/1d4730e99796bf43d873f7ecae738aa6aef90329))
* ensure prod map is created with correct type for SiderealStream ([9d1e773](https://github.com/radiocosmology/draco/commit/9d1e773e1b9d163a9dddd0e0185f20358b566555))
* garbage collect to mitigate memory leak in containers ([407ea4a](https://github.com/radiocosmology/draco/commit/407ea4acdef9f1aaec72d203d68a219c20f16400))
* incorrect/missing output formats in log messages ([559e8a7](https://github.com/radiocosmology/draco/commit/559e8a7cbfca105f374f29f52440aaded250cca8))
* removed errant print statement and changed exception type in io.py ([babfbf2](https://github.com/radiocosmology/draco/commit/babfbf22773932e6a8d9161081da15380b881610))
* stack axis support for MModes ([446b001](https://github.com/radiocosmology/draco/commit/446b0010d8f157e85af0acb1ed01dc501b6147d8)), closes [#24](https://github.com/radiocosmology/draco/issues/24)


### Features

* added versioneer for generating better version numbers ([90cd257](https://github.com/radiocosmology/draco/commit/90cd25793c9403f5565fc767eef5107492bd603b))
* **AccumulateList:** add a new task to bunch inputs up into a list ([332e9f4](https://github.com/radiocosmology/draco/commit/332e9f48fed39a05d25c76f666cbf187cd08c1a6))
* **ComputeSystemSensitivity:** a task to estimate the system sensitivity ([ac3e5ab](https://github.com/radiocosmology/draco/commit/ac3e5ab8dd44cb8d9b22145fd179222ea9c52283))
* **containers:** Add input axis to weight dataset of StaticGainData. ([e6ca2d9](https://github.com/radiocosmology/draco/commit/e6ca2d9423de004c1cc5acb6fd4e80617d3b3ba2))
* **containers:** create SystemSensitivity container ([cc8672c](https://github.com/radiocosmology/draco/commit/cc8672c0c174e411014921b575656c329b367ae5))
* **containers:** add chunking and compression to containers ([12ece28](https://github.com/radiocosmology/draco/commit/12ece28ed2b1f542bbcfc33e899fe76415f7260a))
* **containers:** Add compression to weights datasets and TimeStream. ([9ec6fbc](https://github.com/radiocosmology/draco/commit/9ec6fbc604575082e518e35d87b541b4a362350a))
* **io:** add Truncate task. ([5a7f9f8](https://github.com/radiocosmology/draco/commit/5a7f9f8fb7570567e9cbb8100f9b666a79e0910a))
* **delay:** Add baseline dependent delay cut ([9e797c3](https://github.com/radiocosmology/draco/commit/9e797c3fe98917ecfba4401db34f2b7a868d5a0d))
* **delay:** Add telescope orientation option. ([b52b83b](https://github.com/radiocosmology/draco/commit/b52b83b1699897d41c02ee390101116e9954841a))
* **delay:** Minor documentation clean-up ([34d257f](https://github.com/radiocosmology/draco/commit/34d257fc1a57c9f0cb898975af65a14fd456c083))
* **ExpandProducts:** generate stack maps to match andata containers ([a7046ca](https://github.com/radiocosmology/draco/commit/a7046ca31a3d76a6aaa48ae8129f65b03d19c0b6))
* **flagging:** add task that sets weights below threshold equal to zero ([d29a9e8](https://github.com/radiocosmology/draco/commit/d29a9e8dde236d751dd177f6b433a2598b86aba8))
* **gain:** add tasks SiderealGains and GainStacker ([9afa726](https://github.com/radiocosmology/draco/commit/9afa726f99797984f4d1d6919b8f8eadf0623d82)), closes [#20](https://github.com/radiocosmology/draco/issues/20)
* **GaussianNoise:** Consider redundancy of baselines ([77e2c75](https://github.com/radiocosmology/draco/commit/77e2c7510e525ea8a97b3a1af106cc4141b7da1f))
* **MaskBaselines:** add option to flag short EW baselines. ([6a511aa](https://github.com/radiocosmology/draco/commit/6a511aa1aa20b5e1b3a39b32f6ae1b7307db6653))
* **rfi:** add SumThreshold and SIR calculation functions ([d5dac11](https://github.com/radiocosmology/draco/commit/d5dac116400b60e802eeb2f2b7357ee1f3d91b7e))
* **RFIMask:** add a container for storing RFI masks ([03d4cb8](https://github.com/radiocosmology/draco/commit/03d4cb8be507db5a215719264b86514893855ade))
* **RFIMask:** make destriping optional and turned off by default ([0d426fa](https://github.com/radiocosmology/draco/commit/0d426fac7cd93e4814da212c2f0e3f6427fdfa4d))
* **RFISensitivityMask:** add a new RFI excision task ([4e2dd04](https://github.com/radiocosmology/draco/commit/4e2dd04e2257c94ccbea016b3a4114ef5df59bdd))
* add a basic sidereal random gain task ([5fa639a](https://github.com/radiocosmology/draco/commit/5fa639a067e518a0cae1d5502c558ee9a67c7473))
* stack formed beams on individual sources ([#37](https://github.com/radiocosmology/draco/issues/37)) ([0464013](https://github.com/radiocosmology/draco/commit/0464013ab0db716a4c7afa2aba4b2e00b978c760))
* **flagging:** quick RFI flagging for scattered TV ([c9fa971](https://github.com/radiocosmology/draco/commit/c9fa971655ed97fb75e4e3ea55d0b0db91971caf))
* **io:** add config option to LoadFilesFromParams ([2de500d](https://github.com/radiocosmology/draco/commit/2de500db7f82b738dc8653307a5384cf60ebfc85)), closes [#29](https://github.com/radiocosmology/draco/issues/29)
* **MModeTransform:** set mmax from a supplied Telescope object ([94bed5a](https://github.com/radiocosmology/draco/commit/94bed5aeebd1b048015fefb0fc32e048492a94e4))
* **sidereal:** add properties to `SiderealGrouper` for processing source transits ([49bed81](https://github.com/radiocosmology/draco/commit/49bed8186d162003003128ca914476e7ec34ed2e))
* **task:** add task that deletes whatever is input and collects garbage. ([d575f08](https://github.com/radiocosmology/draco/commit/d575f08bd1965cb80d3d31adfd912c2d79901074))
* **WaitUntil:** a task to forward input only after a requirement is ready ([ebb1d5a](https://github.com/radiocosmology/draco/commit/ebb1d5ae94f765029c209dfc13451407b752a8a8))
* check task output for NaN's/Inf's and log/dump/skip them. ([1341eb8](https://github.com/radiocosmology/draco/commit/1341eb8cf6a6ab6fa23406d5d16943fd890b8011))
* python 3 support ([3cd9680](https://github.com/radiocosmology/draco/commit/3cd9680d910d7bd95b1c031e44bc809a3c19ce6e))


### Performance Improvements

* **DelaySpectrumEstimator:** significant speedups for large datasets ([4df2c5b](https://github.com/radiocosmology/draco/commit/4df2c5b94906a6e534927c8cf6eb09323ef8e523))
* improve regridding speed ([45264a5](https://github.com/radiocosmology/draco/commit/45264a593e50be3458694d21b192471d56c13d42))
* speed ups for the CollateProducts task ([1a1e9c7](https://github.com/radiocosmology/draco/commit/1a1e9c7e47aa48a441c89c1bae675cc5a600f9c4))


### BREAKING CHANGES

* **delay:** DelayFilter now has a `setup` method and `requires` a telescope object.


# [0.2.0] - 2017-06-24

### Features

* MPI aware logging functionality for tasks
* Support for table like data containers
* Routines for delay spectrum estimation and filtering
* KL based foreground filtering
* SVD based foreground filtering
* Quadratic power spectrum estimation
* Task for setting weights based on the radiometer expectation
* Baseline masking task
* Support for reading `ProductManager` instances (from `driftscan`) and updated
  tasks to allow using them in place of `BeamTransfer` or `TransitTelescope`
  instances as task arguments.

### Fixes

* Added a missing routine `_ensure_list`.


## [0.1.0] - 2016-08-11

### Added

* Initial version of `draco` importing code from the CHIME pipeline.
