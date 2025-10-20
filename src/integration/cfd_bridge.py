"""
Prototype utilities for bridging external CFD solver outputs with the singularity detector.

The helpers assume the solver exports an array with shape
    solution[time_index, ... spatial dimensions ...]
and optional arrays for `times` and `grid`.

Supported file formats:
    - .npz (NumPy compressed archives containing `solution`, `times`, `grid`)
    - .npy (single array, requires explicit --times/--grid elsewhere)
    - .h5/.hdf5 (datasets named `solution`, `times`, `grid`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from unstable_singularity_detector import UnstableSingularityDetector


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data}


def _load_hdf5(path: Path) -> Dict[str, np.ndarray]:
    import h5py  # Lazy import to keep dependency optional

    with h5py.File(path, "r") as handle:
        arrays: Dict[str, np.ndarray] = {}
        for key in ("solution", "times", "grid"):
            if key in handle:
                arrays[key] = np.array(handle[key])
        return arrays


def load_cfd_data(path: Path) -> Dict[str, np.ndarray]:
    """Load CFD output from disk into NumPy arrays."""
    ext = path.suffix.lower()
    if ext == ".npz":
        return _load_npz(path)
    if ext == ".npy":
        return {"solution": np.load(path)}
    if ext in {".h5", ".hdf5"}:
        return _load_hdf5(path)

    raise ValueError(f"Unsupported CFD file extension: {ext}")


def _build_uniform_grid(solution: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Create a uniform Cartesian grid if the CFD output did not provide one."""
    spatial_dims = solution.shape[1:]
    grid_components = []
    for size in spatial_dims:
        coords = torch.linspace(-1.0, 1.0, size, dtype=solution.dtype, device=solution.device)
        grid_components.append(coords)
    mesh = torch.meshgrid(*grid_components, indexing="ij")
    return tuple(mesh)


def prepare_cfd_tensors(
    arrays: Dict[str, np.ndarray],
    *,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Convert solver arrays to torch tensors suitable for the detector."""
    if "solution" not in arrays:
        raise ValueError("CFD data must contain a 'solution' array")

    solution_np = arrays["solution"]
    if solution_np.ndim < 2:
        raise ValueError("Solution array must include time and spatial dimensions")

    solution = torch.from_numpy(solution_np).to(torch.float64)
    if device is not None:
        solution = solution.to(device)

    times_np = arrays.get("times")
    if times_np is None:
        times_np = np.arange(solution.shape[0], dtype=np.float64)
    times = torch.from_numpy(times_np).to(solution.dtype)
    if device is not None:
        times = times.to(device)

    grid_np = arrays.get("grid")
    grid_components: Tuple[torch.Tensor, ...]
    if grid_np is not None:
        grid_tensor = torch.from_numpy(grid_np).to(solution.dtype)
        if device is not None:
            grid_tensor = grid_tensor.to(device)
        if grid_tensor.ndim < 2:
            raise ValueError("Grid array must include spatial components")
        grid_components = tuple(grid_tensor[i] for i in range(grid_tensor.shape[0]))
    else:
        grid_components = _build_uniform_grid(solution)

    return solution, times, grid_components


def run_cfd_detection(
    solution: torch.Tensor,
    times: torch.Tensor,
    grid: Tuple[torch.Tensor, ...],
    *,
    detector: Optional[UnstableSingularityDetector] = None,
) -> Sequence:
    """Execute unstable singularity detection on CFD tensors."""
    detector = detector or UnstableSingularityDetector()
    return detector.detect_unstable_singularities(solution, times, grid)


def summarise_detection_results(results: Sequence) -> Dict[str, Iterable]:
    """Summarise detection results for reporting."""
    return {
        "count": len(results),
        "lambda_values": [res.lambda_value for res in results],
        "instability_orders": [res.instability_order for res in results],
        "confidence_scores": [res.confidence_score for res in results],
        "precision": [res.precision_achieved for res in results],
    }


def export_summary(summary: Dict[str, Iterable], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
