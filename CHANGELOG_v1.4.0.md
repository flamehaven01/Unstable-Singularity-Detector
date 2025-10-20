# Changelog – v1.4.0 (2025-10-20)

## Highlights

- **Stage 1 Orchestration Upgrades**
  - Added optional FSDP wrapping and configurable optimiser hooks (Meta/K-FAC) for multi-stage PINN training.
  - Introduced Lightning/Accelerate integration scaffolding so distributed experiments can plug in without modifying the core trainer.

- **Gauss–Newton Enhancements**
  - Added a lightweight `MetaOptimizer` wrapper and a K-FAC preconditioning hook for high-precision solves.
  - Krylov (CG) path now automatically falls back to direct solves with better logging and damping safeguards.

- **PINN Sampling & Hypernets**
  - `resample_interior_points` enables residual-weighted adaptive collocation refresh.
  - Added `LambdaHyperNet` skeleton to support learnable spectral scaling for subsequent stages.

- **Experiment Tracking & Reporting**
  - `ExperimentTracker.generate_pdf_report` provides one-click PDF summaries (WeasyPrint-backed).
  - Residual error certificates can be generated via `ExperimentTracker.generate_residual_certificate` for proof-style logging.
  - Reproduction CI workflow verifies lambda estimates against reference JSON artefacts and uploads plots.

## Notable Changes

| Area | Description |
|------|-------------|
| Training | Stage 1 trainer now auto-configures AMP, optimisers, and distributed wrappers via config toggles. |
| Optimiser | Meta/K-FAC options wired into Gauss–Newton; CG safeguards improved. |
| Sampling | Adaptive interior resampling API exposed under `PINNSolver`. |
| Reporting | PDF report generation, CFD bridge prototype (`scripts/cfd_bridge.py`), and GitHub workflow (`reproduction-ci.yml`) ensure validation stays green. |

## Breaking Changes

- None. All new pathways are behind configuration flags and default to previous behaviour.

## Upgrade Notes

1. Update to `pip install -e .[dev]` to ensure WeasyPrint optional dependency is available for PDF reports.
2. Review `configs/base.yaml` if you want to toggle Lightning/Accelerate or Meta/K-FAC options; defaults remain unchanged.
3. Enable the new Reproduction CI workflow by ensuring `results/reference.json` and `results/experimental.json` exist on the branch.
