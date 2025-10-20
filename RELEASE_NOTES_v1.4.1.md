# Release Notes - Unstable Singularity Detector v1.4.1

## Summary

v1.4.1 is a maintenance patch that stabilises the enhanced Gauss-Newton optimiser when sampling residual subsets across CPU and GPU runs.

## Fixes

- Ensure the rank-1 Hessian sampler builds `torch.long` index tensors on the same device as the Jacobian to avoid mixed-device indexing failures.
- Scale Hessian-vector estimates by the actual sample count so curvature magnitudes stay accurate even when fewer residuals are available.

## Documentation

- Updated the README with a v1.4.1 patch highlight and refreshed the version string.
- Documented the release in the changelog and added these notes for discoverability.

## Testing

- `pytest tests/test_gauss_newton_enhanced.py -k rank1`
