#!/usr/bin/env python3
"""Prototype CLI for linking external CFD outputs with the unstable singularity detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from integration import cfd_bridge
from unstable_singularity_detector import UnstableSingularityDetector
from utils.certificates import build_residual_certificate, save_residual_certificate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector on CFD solver output")
    parser.add_argument("input", type=Path, help="Path to CFD output (.npz/.npy/.h5)")
    parser.add_argument("--equation", default="euler_3d", help="Equation type for detector")
    parser.add_argument("--precision-target", type=float, default=1e-14, help="Residual tolerance target")
    parser.add_argument("--output", type=Path, help="Optional JSON summary output path")
    parser.add_argument("--certificate", action="store_true", help="Create residual certificate when possible")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Execution device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    arrays = cfd_bridge.load_cfd_data(args.input)
    solution, times, grid = cfd_bridge.prepare_cfd_tensors(arrays, device=device)

    detector = UnstableSingularityDetector(
        equation_type=args.equation,
        precision_target=args.precision_target,
    )
    results = cfd_bridge.run_cfd_detection(solution, times, grid, detector=detector)
    summary = cfd_bridge.summarise_detection_results(results)

    print(f"Detected {summary['count']} candidate singularities")
    if summary['count']:
        print("Lambda estimates:", summary['lambda_values'])
        print("Instability orders:", summary['instability_orders'])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary saved to {args.output}")

    if args.certificate and summary['count']:
        loss_history = arrays.get("loss_history")
        final_loss = arrays.get("final_loss")
        if not loss_history and final_loss is None:
            loss_history = [summary['precision'][0] ** 2]

        certificate = build_residual_certificate(
            loss_history=loss_history or [final_loss or 1.0],
            tolerance=args.precision_target,
            metadata={"lambda_values": summary['lambda_values']},
        )
        paths = save_residual_certificate(certificate, Path("certificates"), base_name="cfd_detection")
        print("Certificate saved:", paths)


if __name__ == "__main__":
    main()
