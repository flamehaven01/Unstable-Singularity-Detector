import math
from pathlib import Path

import pytest

from src.utils.certificates import (
    build_residual_certificate,
    render_residual_certificate_md,
    save_residual_certificate,
)


def test_residual_certificate_satisfied(tmp_path: Path):
    loss_history = [0.5 * (2.0e-6) ** 2, 0.5 * (5.0e-7) ** 2]
    certificate = build_residual_certificate(
        loss_history,
        tolerance=1.0e-6,
        safety_factor=0.1,
    )

    assert certificate.holds
    assert math.isclose(certificate.final_residual_l2, 5.0e-7, rel_tol=1e-9)
    assert certificate.metadata["margin_satisfied"] is True

    markdown = render_residual_certificate_md(certificate)
    assert "Residual Error Certificate" in markdown

    paths = save_residual_certificate(certificate, tmp_path, base_name="proof")
    assert paths["markdown"].exists()
    assert paths["json"].exists()


def test_residual_certificate_violation():
    loss_history = [0.5 * (2.5e-6) ** 2]
    certificate = build_residual_certificate(loss_history, tolerance=1.0e-6)
    assert certificate.holds is False
    assert certificate.metadata["margin_satisfied"] is False
    assert certificate.safety_margin == pytest.approx(-1.5e-6)
