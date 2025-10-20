# Release Notes – Unstable Singularity Detector v1.4.0

## Summary

v1.4.0 focuses on hardening the multi-stage PINN pipeline, extending the Gauss–Newton optimiser, and adding reproducibility tooling so validation stays automated.

## Key Features

- **Stage 1 Training Orchestrator**
  - Optional FSDP wrapping for large models.
  - Lightning and Accelerate hooks to prototype distributed runs without rewriting training code.
  - Config-driven optimiser selection with Meta/K-FAC options.

- **Gauss–Newton Improvements**
  - Meta-optimiser wrapper and K-FAC preconditioning hook.
  - Safer Krylov (CG) fallback handling and richer history logging (loss, damping, learning rate).

- **Adaptive Sampling & Hypernets**
  - Residual-weighted interior resampling (`PINNSolver.resample_interior_points`).
  - `LambdaHyperNet` stub prepares spectral scaling for future stage-2 refinements.

- **Reporting & CI**
  - `ExperimentTracker.generate_pdf_report` turns MLflow runs into shareable PDFs.
  - Residual certificates (`generate_residual_certificate`) summarise residual bounds for proof-oriented workflows.
  - New reproduction CI workflow validates lambda estimates vs `results/reference.json` and uploads plots.

## Upgrade Checklist

1. `pip install -e ".[dev]"` and add WeasyPrint if you plan to generate PDF reports.
2. Keep `results/reference.json` + `results/experimental.json` updated for the reproduction workflow.
3. Review `MultiStageConfig` for newly added toggles (`use_fsdp`, `use_lightning`, `use_accelerate`, `optimizer`).

## Compatibility

- No breaking API changes; defaults preserve 1.3.x behaviour.
- Optional dependencies (Lightning, Accelerate, WeasyPrint, K-FAC) remain opt-in.

## Testing

- `pytest tests/test_multistage_training.py`
- `scripts/replicate_metrics.py --ref results/reference.json --exp results/experimental.json --rtol 1e-3`
