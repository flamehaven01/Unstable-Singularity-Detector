"""Integration utilities for external solver outputs."""

from .cfd_bridge import (
    load_cfd_data,
    prepare_cfd_tensors,
    run_cfd_detection,
    summarise_detection_results,
)

__all__ = [
    "load_cfd_data",
    "prepare_cfd_tensors",
    "run_cfd_detection",
    "summarise_detection_results",
]
