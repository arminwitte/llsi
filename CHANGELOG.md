# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-12-28
### Added
- MATLAB System Identification Toolbox wrapper API (`llsi.matlab`) for easier migration from MATLAB.
- Pandas interoperability: `SysIdData.to_pandas()` and `SysIdData.from_pandas()`.
- `pandas` dependency.
- `uv.lock` is now tracked in the repository for reproducible builds.

## [0.4.0] - 2025-12-27
### Added
- Initial changelog creation.
- Support for `hatchling` build system.
- `numba` acceleration for state space evaluation.
- `tqdm` for progress bars.

### Changed
- Updated project structure to standard Python package layout.
