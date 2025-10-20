#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

from integration.cfd_bridge import prepare_cfd_tensors, run_cfd_detection, summarise_detection_results
from unstable_singularity_detector import UnstableSingularityDetector


def create_synthetic_cfd_data(dim=3):
    time_steps = 5
    grid_size = 6
    coords = np.linspace(-1.0, 1.0, grid_size)
    mesh = np.meshgrid(*([coords] * dim), indexing="ij")
    grid = np.stack(mesh, axis=0)

    times = np.linspace(0.7, 0.95, time_steps)
    solution = []
    for t in times:
        field = np.exp(-np.sum([m ** 2 for m in mesh], axis=0)) * (1.0 - t) ** (-1.1)
        solution.append(field)
    solution = np.stack(solution, axis=0)
    return {"solution": solution, "times": times, "grid": grid}


def test_prepare_cfd_tensors():
    arrays = create_synthetic_cfd_data()
    solution, times, grid = prepare_cfd_tensors(arrays)
    assert isinstance(solution, torch.Tensor)
    assert solution.ndimension() == 4
    assert len(grid) == 3


def test_run_cfd_detection_smoke():
    arrays = create_synthetic_cfd_data()
    solution, times, grid = prepare_cfd_tensors(arrays)
    detector = UnstableSingularityDetector(max_instability_order=2)
    results = run_cfd_detection(solution, times, grid, detector=detector)
    summary = summarise_detection_results(results)
    assert "count" in summary
    assert isinstance(summary["lambda_values"], list)
