"""Utility modules for unstable singularity detection."""

from .metrics import pde_residual_stats
from .checks import assert_finite
from .repro import set_global_seed
from .certificates import (
    ResidualCertificate,
    build_residual_certificate,
    render_residual_certificate_md,
    save_residual_certificate,
)

__all__ = [
    "ResidualCertificate",
    "build_residual_certificate",
    "render_residual_certificate_md",
    "save_residual_certificate",
    "pde_residual_stats",
    "assert_finite",
    "set_global_seed",
]
