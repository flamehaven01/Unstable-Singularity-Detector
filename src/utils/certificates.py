"""Utilities for generating residual error certificates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


def _to_float_sequence(values: Sequence[Any]) -> Iterable[float]:
    for value in values:
        yield float(value)


@dataclass(frozen=True)
class ResidualCertificate:
    """Structured summary of residual error bounds."""

    created_at: str
    tolerance: float
    final_loss: float
    final_residual_l2: float
    max_loss: float
    max_residual_l2: float
    iterations: int
    final_gradient_norm: Optional[float]
    max_gradient_norm: Optional[float]
    safety_margin: float
    holds: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_residual_certificate(
    loss_history: Sequence[Any],
    *,
    tolerance: float,
    residual_history: Optional[Sequence[Any]] = None,
    gradient_history: Optional[Sequence[Any]] = None,
    safety_factor: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ResidualCertificate:
    if not loss_history:
        raise ValueError("loss_history must contain at least one entry")

    loss_values = [max(float(loss), 0.0) for loss in loss_history]
    final_loss = loss_values[-1]
    max_loss = max(loss_values)

    if residual_history:
        residual_values = [abs(float(val)) for val in residual_history]
    else:
        residual_values = [math.sqrt(max(0.0, 2.0 * loss)) for loss in loss_values]

    final_residual = residual_values[-1]
    max_residual = max(residual_values)

    grad_values = [abs(float(val)) for val in gradient_history] if gradient_history else []
    final_grad = grad_values[-1] if grad_values else None
    max_grad = max(grad_values) if grad_values else None

    tolerance = float(tolerance)
    safety_margin = tolerance - final_residual
    required_margin = tolerance * max(float(safety_factor), 0.0)
    margin_satisfied = safety_margin >= required_margin
    holds = final_residual <= tolerance

    cert_metadata = dict(metadata or {})
    cert_metadata.setdefault("safety_factor", float(safety_factor))
    cert_metadata.setdefault("margin_requirement", required_margin)
    cert_metadata.setdefault("margin_satisfied", margin_satisfied)

    return ResidualCertificate(
        created_at=datetime.now(UTC).isoformat(),
        tolerance=tolerance,
        final_loss=final_loss,
        final_residual_l2=final_residual,
        max_loss=max_loss,
        max_residual_l2=max_residual,
        iterations=len(loss_values),
        final_gradient_norm=final_grad,
        max_gradient_norm=max_grad,
        safety_margin=safety_margin,
        holds=holds,
        metadata=cert_metadata,
    )


def render_residual_certificate_md(cert: ResidualCertificate) -> str:
    lines = [
        "# Residual Error Certificate",
        "",
        f"- Generated: {cert.created_at}",
        f"- Residual tolerance: {cert.tolerance:.3e}",
        f"- Final residual (L2): {cert.final_residual_l2:.3e}",
        f"- Maximum residual (L2): {cert.max_residual_l2:.3e}",
        f"- Final loss: {cert.final_loss:.3e}",
        f"- Maximum loss: {cert.max_loss:.3e}",
        f"- Iterations: {cert.iterations}",
        f"- Final gradient norm: "
        f"{cert.final_gradient_norm:.3e}" if cert.final_gradient_norm is not None else "- Final gradient norm: n/a",
        f"- Maximum gradient norm: "
        f"{cert.max_gradient_norm:.3e}" if cert.max_gradient_norm is not None else "- Maximum gradient norm: n/a",
        "",
    ]

    status = "SATISFIED" if cert.holds else "VIOLATED"
    lines.append(f"## Inequality Status: **{status}**")
    if cert.final_loss >= 0.0:
        bound = math.sqrt(max(0.0, 2.0 * cert.final_loss))
        lines.append(
            f"\\[ \\|r\\|_2 \\le \\sqrt{{2\\,\\mathcal{{L}}}} "
            f"= \\sqrt{{2 \\cdot {cert.final_loss:.3e}}} = {bound:.3e} "
            f"\\le {cert.tolerance:.3e} \\]"
        )
    lines.append("")

    margin_requirement = cert.metadata.get("margin_requirement", 0.0)
    margin_status = "SATISFIED" if cert.metadata.get("margin_satisfied", cert.holds) else "VIOLATED"
    lines.extend(
        [
            "## Safety Margin",
            f"- Safety factor: {cert.metadata.get('safety_factor', 0.0):.3f}",
            f"- Required margin: {margin_requirement:.3e}",
            f"- Achieved margin: {cert.safety_margin:.3e}",
            f"- Margin status: **{margin_status}**",
            "",
        ]
    )

    if cert.metadata:
        lines.append("## Metadata")
        for key, value in sorted(cert.metadata.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append(
        r"Assumption: $\mathcal{L} = \tfrac{1}{2}\|r\|_2^2$ holds for the provided optimisation result."
    )

    return "\n".join(lines)


def save_residual_certificate(
    cert: ResidualCertificate,
    output_dir: Path,
    *,
    base_name: str = "residual_certificate",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{base_name}.md"
    json_path = output_dir / f"{base_name}.json"

    md_path.write_text(render_residual_certificate_md(cert), encoding="utf-8")
    json_path.write_text(json.dumps(cert.to_dict(), indent=2), encoding="utf-8")

    return {"markdown": md_path, "json": json_path}


__all__ = [
    "ResidualCertificate",
    "build_residual_certificate",
    "render_residual_certificate_md",
    "save_residual_certificate",
]
