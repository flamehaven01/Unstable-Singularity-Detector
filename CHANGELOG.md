# Unstable Singularity Detector Changelog

## [Unreleased]

- TBD

## [1.4.0] - 2025-10-20

### Highlights

- Stage 1 orchestration upgrades with optional FSDP and optimiser hooks (Meta/K-FAC).
- Gauss–Newton enhancements (MetaOptimizer, K-FAC, improved Krylov fallback).
- Residual certificates and reproduction CI for verifiable proofs.
- CFD bridge CLI linking external solver snapshots to detection.

### Added

- Residual certificate utilities (`src/utils/certificates.py`) and ExperimentTracker integration.
- CFD bridge helpers (`src/integration/cfd_bridge.py`) and CLI script (`scripts/cfd_bridge.py`).
- 3D detection tests for spatial profile extraction and lambda estimation.

### Fixed

- Gradient scanning skips negative time indices; residuals and instabilities support N-D tensors.

### Changed

- README updated with 1.4.0 highlights and CFD workflow.

## [1.3.2] - 2025-10-04

- Enhanced Gauss–Newton logging and residual fixes.
- Torch shim utilities for reproducibility testing.

## [1.3.1] - 2025-10-03

- Initial public release notes summarising Phase A/B/C completions.

## [History prior to 1.3.1]

- Refer to project documentation for earlier milestones.
