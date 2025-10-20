# Unstable Singularity Detector Changelog

## [Unreleased]

- TBD

## [1.4.1] - 2025-10-20

### Fixed

- Rank-1 Hessian sampler now creates device-aware `torch.long` indices, preventing CUDA indexing errors.
- Curvature scaling uses the actual sample count so Hessian-vector products stay consistent on small batches.

### Documentation

- README updated with the v1.4.1 patch summary and version bump.
- Added release notes for v1.4.1.

### Testing

- `pytest tests/test_gauss_newton_enhanced.py -k rank1`

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
