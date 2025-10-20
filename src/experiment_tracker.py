#!/usr/bin/env python3
"""
Experiment Tracking System

MLflow-based experiment tracking for reproducible research.
Automatic logging of parameters, metrics, artifacts, and models.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from pathlib import Path
import json
import pickle
import numpy as np
import torch
from datetime import datetime
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import h5py

from utils.certificates import build_residual_certificate, save_residual_certificate

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Comprehensive experiment tracking system

    Features:
    - Automatic parameter and metric logging
    - Model artifact management
    - Visualization and plot tracking
    - Data versioning integration
    - Experiment comparison and analysis
    """

    def __init__(self,
                 experiment_name: str = "unstable_singularity_detection",
                 tracking_uri: Optional[str] = None,
                 artifact_location: Optional[str] = None):
        """
        Initialize experiment tracker

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Base location for artifacts
        """
        self.experiment_name = experiment_name

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local file-based tracking by default
            tracking_dir = Path("./mlruns").resolve()
            tracking_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{tracking_dir}")

        # Set or create experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

        self.client = MlflowClient()
        self.active_run = None

        logger.info(f"Initialized experiment tracker: {experiment_name}")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    def start_run(self,
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None,
                  nested: bool = False) -> str:
        """Start a new MLflow run

        Args:
            run_name: Name for the run
            tags: Dictionary of tags to apply
            nested: Whether this is a nested run

        Returns:
            Run ID
        """
        if tags is None:
            tags = {}

        # Add default tags
        default_tags = {
            "mlflow.runName": run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "project": "unstable-singularity-detector",
            "version": "1.0.0"
        }
        tags.update(default_tags)

        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
            nested=nested
        )

        run_id = self.active_run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

        return run_id

    def end_run(self):
        """End the current MLflow run"""
        if self.active_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
            self.active_run = None

    def log_config(self, config: Union[DictConfig, Dict[str, Any]], prefix: str = ""):
        """Log configuration parameters

        Args:
            config: Configuration object or dictionary
            prefix: Prefix for parameter names
        """
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        def flatten_dict(d, parent_key='', sep='.'):
            """Flatten nested dictionary"""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # Flatten and log parameters
        flat_config = flatten_dict(config)

        for key, value in flat_config.items():
            param_name = f"{prefix}.{key}" if prefix else key
            try:
                # Convert to string if not a basic type
                if not isinstance(value, (str, int, float, bool)):
                    value = str(value)
                mlflow.log_param(param_name, value)
            except Exception as e:
                logger.warning(f"Failed to log parameter {param_name}: {e}")

        logger.info(f"Logged {len(flat_config)} configuration parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics with optional step

        Args:
            metrics: Dictionary of metric name -> value
            step: Step number for the metrics
        """
        for name, value in metrics.items():
            try:
                mlflow.log_metric(name, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {name}: {e}")

    def log_training_metrics(self,
                           epoch: int,
                           loss_dict: Dict[str, float],
                           additional_metrics: Optional[Dict[str, float]] = None):
        """Log training metrics for a specific epoch

        Args:
            epoch: Current epoch number
            loss_dict: Dictionary of loss components
            additional_metrics: Additional metrics to log
        """
        all_metrics = loss_dict.copy()
        if additional_metrics:
            all_metrics.update(additional_metrics)

        self.log_metrics(all_metrics, step=epoch)

    def log_model(self,
                  model: torch.nn.Module,
                  artifact_path: str = "model",
                  model_name: Optional[str] = None,
                  signature=None):
        """Log PyTorch model

        Args:
            model: PyTorch model to log
            artifact_path: Path within the run's artifact directory
            model_name: Registered model name
            signature: Model signature
        """
        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=model_name,
                signature=signature
            )
            logger.info(f"Logged model to artifact path: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_figure(self,
                   figure: plt.Figure,
                   filename: str,
                   artifact_path: str = "figures"):
        """Log matplotlib figure

        Args:
            figure: Matplotlib figure
            filename: Filename for the figure
            artifact_path: Artifact directory path
        """
        try:
            # Create temporary directory
            temp_dir = Path("./temp_artifacts")
            temp_dir.mkdir(exist_ok=True)

            fig_path = temp_dir / filename
            figure.savefig(fig_path, dpi=300, bbox_inches='tight')

            mlflow.log_artifact(str(fig_path), artifact_path)

            # Cleanup
            fig_path.unlink()

            logger.info(f"Logged figure: {filename}")
        except Exception as e:
            logger.error(f"Failed to log figure {filename}: {e}")

    def log_data_artifact(self,
                         data: Union[np.ndarray, torch.Tensor, Dict],
                         filename: str,
                         artifact_path: str = "data",
                         format: str = "auto"):
        """Log data artifacts

        Args:
            data: Data to save (numpy array, tensor, or dict)
            filename: Filename for the artifact
            artifact_path: Artifact directory path
            format: Data format (auto, hdf5, pickle, numpy)
        """
        try:
            temp_dir = Path("./temp_artifacts")
            temp_dir.mkdir(exist_ok=True)

            file_path = temp_dir / filename

            # Auto-detect format
            if format == "auto":
                if filename.endswith(('.h5', '.hdf5')):
                    format = "hdf5"
                elif filename.endswith('.pkl'):
                    format = "pickle"
                elif filename.endswith('.npy'):
                    format = "numpy"
                else:
                    format = "pickle"

            # Save data based on format
            if format == "hdf5":
                with h5py.File(file_path, 'w') as f:
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, torch.Tensor):
                                value = value.detach().cpu().numpy()
                            f.create_dataset(key, data=value)
                    else:
                        if isinstance(data, torch.Tensor):
                            data = data.detach().cpu().numpy()
                        f.create_dataset('data', data=data)

            elif format == "numpy":
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                np.save(file_path, data)

            elif format == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)

            # Log to MLflow
            mlflow.log_artifact(str(file_path), artifact_path)

            # Cleanup
            file_path.unlink()

            logger.info(f"Logged data artifact: {filename}")

        except Exception as e:
            logger.error(f"Failed to log data artifact {filename}: {e}")

    def log_singularity_results(self,
                               results: List,
                               artifact_path: str = "results"):
        """Log singularity detection results

        Args:
            results: List of SingularityDetectionResult objects
            artifact_path: Artifact directory path
        """
        if not results:
            logger.warning("No singularity results to log")
            return

        try:
            # Convert results to serializable format
            serializable_results = []
            metrics_summary = {
                'num_singularities': len(results),
                'avg_lambda': 0.0,
                'avg_confidence': 0.0,
                'avg_precision': 0.0
            }

            lambda_values = []
            confidences = []
            precisions = []

            for i, result in enumerate(results):
                result_dict = {
                    'index': i,
                    'singularity_type': result.singularity_type.value,
                    'lambda_value': float(result.lambda_value),
                    'instability_order': int(result.instability_order),
                    'confidence_score': float(result.confidence_score),
                    'time_to_blowup': float(result.time_to_blowup),
                    'residual_error': float(result.residual_error),
                    'precision_achieved': float(result.precision_achieved)
                }
                serializable_results.append(result_dict)

                lambda_values.append(result.lambda_value)
                confidences.append(result.confidence_score)
                precisions.append(result.precision_achieved)

            # Calculate summary metrics
            if lambda_values:
                metrics_summary.update({
                    'avg_lambda': float(np.mean(lambda_values)),
                    'std_lambda': float(np.std(lambda_values)),
                    'min_lambda': float(np.min(lambda_values)),
                    'max_lambda': float(np.max(lambda_values)),
                    'avg_confidence': float(np.mean(confidences)),
                    'avg_precision': float(np.mean(precisions)),
                    'best_precision': float(np.min(precisions))
                })

            # Log summary metrics
            self.log_metrics(metrics_summary)

            # Log detailed results
            self.log_data_artifact(
                serializable_results,
                "singularity_results.json",
                artifact_path,
                format="pickle"
            )

            logger.info(f"Logged {len(results)} singularity detection results")

        except Exception as e:
            logger.error(f"Failed to log singularity results: {e}")

    def replay_run(self, run_id: str, output_dir: Union[str, Path] = "replay") -> Optional[Dict[str, Any]]:
        """
        Restore configuration and artifacts from a previous MLflow run.

        Args:
            run_id: Identifier of the MLflow run to restore.
            output_dir: Local directory for downloaded artifacts.

        Returns:
            Summary dictionary containing params, metrics, and artifact path if successful.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            run = self.client.get_run(run_id)

            def _download_artifacts(prefix: str = "") -> None:
                artifacts = self.client.list_artifacts(run_id, prefix)
                for artifact in artifacts:
                    if artifact.is_dir:
                        _download_artifacts(artifact.path)
                    else:
                        local_file = self.client.download_artifacts(run_id, artifact.path, dst_path=output_path)
                        logger.info(f"[Replay] Downloaded artifact {artifact.path} -> {local_file}")

            _download_artifacts()

            replay_summary = {
                "run_id": run_id,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "artifacts_dir": str(output_path.resolve())
            }
            logger.info(f"[Replay] Restored run {run_id} to {replay_summary['artifacts_dir']}")
            return replay_summary

        except Exception as exc:
            logger.error(f"[Replay] Failed to restore run {run_id}: {exc}")
            return None

    def compare_runs(self,
                    run_ids: List[str],
                    metrics: Optional[List[str]] = None) -> Dict:
        """Compare multiple runs

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare

        Returns:
            Comparison data dictionary
        """
        comparison_data = {}

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'params': run.data.params,
                    'metrics': run.data.metrics
                }
                comparison_data[run_id] = run_data

            except Exception as e:
                logger.error(f"Failed to get data for run {run_id}: {e}")

        return comparison_data

    def get_best_run(self, metric_name: str, direction: str = "min", auto_link: bool = False) -> Optional[str]:
        """Get the best run based on a metric

        Args:
            metric_name: Name of the metric to optimize
            direction: 'min' or 'max'
            auto_link: If True, set this run as the active context for next stage (Patch #2.3)

        Returns:
            Best run ID or None
        """
        try:
            experiment = mlflow.get_experiment(self.experiment_id)
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string="",
                order_by=[f"metrics.{metric_name} {'ASC' if direction == 'min' else 'DESC'}"]
            )

            if not runs.empty:
                best_run_id = runs.iloc[0]['run_id']
                logger.info(f"Best run for {metric_name}: {best_run_id}")

                # Auto-link best run as active (Patch #2.3)
                if auto_link:
                    self.active_run = self.client.get_run(best_run_id)
                    logger.info(f"[Auto-Link] Linked best run {best_run_id} as active for next stage")

                return best_run_id

        except Exception as e:
            logger.error(f"Failed to find best run: {e}")

        return None

    def cleanup_temp_artifacts(self):
        """Clean up temporary artifact directory"""
        temp_dir = Path("./temp_artifacts")
        if temp_dir.exists():
            for file_path in temp_dir.iterdir():
                file_path.unlink()
            temp_dir.rmdir()

    # Patch #7.2: Config Hash Tracking
    def log_config_hash(self, cfg: dict):
        """
        Log SHA1 hash of the config dict for reproducibility

        Args:
            cfg: Configuration dictionary
        """
        import hashlib
        import json

        cfg_str = json.dumps(cfg, sort_keys=True)
        cfg_hash = hashlib.sha1(cfg_str.encode()).hexdigest()

        self.client.log_param(self.active_run.info.run_id, "config_hash", cfg_hash)
        logger.info(f"[Config] Logged config hash={cfg_hash[:8]}")

        return cfg_hash

    # Patch #7.3: Run Provenance
    def log_provenance(self):
        """
        Log code + environment provenance (git commit, hostname, seed)
        """
        import subprocess
        import socket
        import random

        # Git commit
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            commit = "unknown"

        # Hostname
        hostname = socket.gethostname()

        # Random seed (set and log)
        seed = random.randint(0, 10**6)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Log all provenance info
        self.client.log_param(self.active_run.info.run_id, "git_commit", commit)
        self.client.log_param(self.active_run.info.run_id, "hostname", hostname)
        self.client.log_param(self.active_run.info.run_id, "random_seed", seed)

        logger.info(f"[Provenance] commit={commit[:8]}, host={hostname}, seed={seed}")

        return {"commit": commit, "hostname": hostname, "seed": seed}

    def generate_pdf_report(self,
                            run_id: str,
                            output_file: str = "experiment_report.pdf") -> Optional[str]:
        """
        Generate a lightweight PDF experiment summary.

        Requires WeasyPrint and its system dependencies to be installed; otherwise a warning is raised.
        """
        try:
            from weasyprint import HTML  # type: ignore
        except ImportError:
            logger.error("[Report] WeasyPrint is not installed. Skipping PDF report generation.")
            return None

        run = self.client.get_run(run_id)
        params = json.dumps(run.data.params, indent=2, default=str)
        metrics = json.dumps(run.data.metrics, indent=2, default=str)

        html = f"""
        <html>
          <head><meta charset='utf-8'><title>Experiment Report</title></head>
          <body>
            <h1>Experiment Report</h1>
            <h2>Run ID: {run_id}</h2>
            <h3>Parameters</h3>
            <pre>{params}</pre>
            <h3>Metrics</h3>
            <pre>{metrics}</pre>
          </body>
        </html>
        """

        try:
            HTML(string=html).write_pdf(output_file)
            logger.info(f"[Report] PDF saved to {output_file}")
            return output_file
        except Exception as exc:
            logger.error(f"[Report] Failed to generate PDF: {exc}")
            return None

    def generate_residual_certificate(
        self,
        optimization_result: Dict[str, Any],
        tolerance: float,
        *,
        residual_history: Optional[Sequence[Any]] = None,
        safety_factor: float = 0.0,
        output_dir: Union[str, Path] = "certificates",
        filename: str = "residual_certificate",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Build and persist a residual error certificate for an optimisation run.

        Args:
            optimization_result: Output dictionary returned by the optimiser.
            tolerance: Residual tolerance the run is expected to satisfy.
            residual_history: Optional sequence of residual norms per iteration.
            safety_factor: Fractional safety margin (0.1 means 10% slack).
            output_dir: Directory used to store certificate artefacts.
            filename: Base filename (without extension) for the certificate files.
            metadata: Optional extra metadata to embed in the certificate.

        Returns:
            Tuple of (ResidualCertificate, dict with output paths).
        """
        loss_history = optimization_result.get("loss_history") or []
        if not loss_history and "loss" in optimization_result:
            loss_history = [optimization_result["loss"]]
        if not loss_history:
            raise ValueError("optimization_result must provide loss_history or loss")

        gradient_history = optimization_result.get("gradient_norm_history") or []
        residual_history = residual_history or optimization_result.get("residual_history")

        meta = dict(metadata or {})
        meta.setdefault("iterations_reported", optimization_result.get("iterations"))
        if optimization_result.get("gradient_norm") is not None:
            meta.setdefault("final_gradient_norm", optimization_result.get("gradient_norm"))

        certificate = build_residual_certificate(
            loss_history=loss_history,
            tolerance=tolerance,
            residual_history=residual_history,
            gradient_history=gradient_history,
            safety_factor=safety_factor,
            metadata=meta,
        )

        output_dir = Path(output_dir)
        paths = save_residual_certificate(certificate, output_dir, base_name=filename)
        logger.info(
            "[Certificate] Residual certificate stored at %s (holds=%s)",
            output_dir,
            certificate.holds,
        )

        if self.active_run:
            artifact_folder = str(output_dir.name)
            for artifact_path in paths.values():
                mlflow.log_artifact(str(artifact_path), artifact_path=artifact_folder)

        return certificate, paths

    # Patch #9.4: Markdown Summary
    def summarize_run(self, run_id: str = None, output_file: str = "experiment_summary.md"):
        """
        Generate Markdown summary of experiment

        Args:
            run_id: MLflow run ID (if None, use active run)
            output_file: Output markdown file path
        """
        if run_id is None:
            if self.active_run is None:
                logger.error("No active run and no run_id provided")
                return
            run_id = self.active_run.info.run_id

        run = self.client.get_run(run_id)
        params = run.data.params
        metrics = run.data.metrics

        # Generate markdown
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Experiment Summary\n\n")
            f.write(f"**Run ID**: {run_id}\n")
            f.write(f"**Run Name**: {run.info.run_name}\n")
            f.write(f"**Start Time**: {datetime.fromtimestamp(run.info.start_time/1000).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Status**: {run.info.status}\n\n")

            f.write("## Parameters\n\n")
            for k, v in sorted(params.items()):
                f.write(f"- **{k}**: {v}\n")

            f.write("\n## Metrics\n\n")
            for k, v in sorted(metrics.items()):
                f.write(f"- **{k}**: {v}\n")

            f.write("\n---\n")
            f.write(f"*Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"[Summary] Markdown summary saved to {output_file}")
        return output_file

    def track_dataset(self, dataset_path: str):
        """
        Track dataset version using hash (compatible with DVC/Git) (Patch #7.1)

        Args:
            dataset_path: Path to dataset file to track

        Returns:
            SHA1 hash of the dataset
        """
        if self.active_run is None:
            logger.error("No active run. Call start_run() first.")
            return None

        try:
            import hashlib

            BUF_SIZE = 65536
            sha1 = hashlib.sha1()

            with open(dataset_path, "rb") as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    sha1.update(data)

            dataset_hash = sha1.hexdigest()

            self.client.log_param(self.active_run.info.run_id, "dataset_path", dataset_path)
            self.client.log_param(self.active_run.info.run_id, "dataset_hash", dataset_hash)

            logger.info(f"[Dataset] Tracked {dataset_path} with hash={dataset_hash[:8]}...")
            return dataset_hash

        except FileNotFoundError:
            logger.error(f"[Dataset] File not found: {dataset_path}")
            return None
        except Exception as e:
            logger.error(f"[Dataset] Failed to track: {e}")
            return None

    def replay_metadata(self, run_id: str):
        """
        Rebuild experiment setup from logged metadata (Patch #7.4)

        Args:
            run_id: MLflow run ID to replay

        Returns:
            Dictionary of run parameters
        """
        try:
            run = self.client.get_run(run_id)
            params = run.data.params

            logger.info(f"[Replay Metadata] Run {run_id}")
            logger.info("[Replay Metadata] Run parameters:")
            for k, v in params.items():
                logger.info(f"  {k}: {v}")

            return params

        except Exception as e:
            logger.error(f"[Replay Metadata] Failed: {e}")
            return {}

    def export_notebook(self, run_id: str, filename: str = "analysis.ipynb"):
        """
        Generate Jupyter notebook with experiment analysis cells (Patch #9.3)

        Args:
            run_id: MLflow run ID
            filename: Output notebook file path

        Returns:
            Path to generated notebook
        """
        try:
            import nbformat as nbf

            run = self.client.get_run(run_id)
            params = run.data.params
            metrics = run.data.metrics

            # Create new notebook
            nb = nbf.v4.new_notebook()

            # Title cell
            nb.cells.append(nbf.v4.new_markdown_cell(
                f"# Experiment Analysis Notebook\n\n"
                f"**Run ID**: `{run_id}`\n\n"
                f"**Status**: {run.info.status}\n\n"
                f"Generated automatically from MLflow tracking."
            ))

            # Parameters cell
            params_code = "# Experiment Parameters\n"
            params_code += f"params = {params}\n"
            params_code += "print('Parameters:')\n"
            params_code += "for k, v in params.items():\n"
            params_code += "    print(f'  {k}: {v}')"
            nb.cells.append(nbf.v4.new_code_cell(params_code))

            # Metrics cell
            metrics_code = "# Experiment Metrics\n"
            metrics_code += f"metrics = {metrics}\n"
            metrics_code += "print('Metrics:')\n"
            metrics_code += "for k, v in metrics.items():\n"
            metrics_code += "    print(f'  {k}: {v}')"
            nb.cells.append(nbf.v4.new_code_cell(metrics_code))

            # Visualization template
            viz_code = """# Custom Analysis and Visualization
import matplotlib.pyplot as plt
import numpy as np

# Add your custom plots here
# Example:
# plt.figure(figsize=(10, 6))
# plt.plot(residual_history)
# plt.yscale('log')
# plt.title('Convergence')
# plt.show()"""
            nb.cells.append(nbf.v4.new_code_cell(viz_code))

            # Write notebook
            with open(filename, "w", encoding="utf-8") as f:
                nbf.write(nb, f)

            logger.info(f"[Notebook Export] Jupyter notebook saved to {filename}")
            return filename

        except ImportError:
            logger.error("[Notebook Export] nbformat not installed. Install: pip install nbformat")
            return None
        except Exception as e:
            logger.error(f"[Notebook Export] Failed: {e}")
            return None

    def track_lambda_timeseries(self, lambdas: list, timestamps: list = None):
        """
        Track lambda instability values over time (Phase C - Interactive λ analysis)

        Args:
            lambdas: List of lambda values
            timestamps: Optional list of timestamps (if None, uses indices)
        """
        if self.active_run is None:
            logger.error("No active run. Call start_run() first.")
            return

        try:
            import numpy as np

            if timestamps is None:
                timestamps = list(range(len(lambdas)))

            # Log statistics
            lambda_stats = {
                "lambda_mean": float(np.mean(lambdas)),
                "lambda_std": float(np.std(lambdas)),
                "lambda_min": float(np.min(lambdas)),
                "lambda_max": float(np.max(lambdas)),
                "lambda_count": len(lambdas)
            }

            self.log_metrics(lambda_stats)

            # Log timeseries data
            lambda_data = {
                "timestamps": timestamps,
                "lambdas": lambdas,
                "statistics": lambda_stats
            }

            self.log_data_artifact(
                lambda_data,
                "lambda_timeseries.json",
                artifact_path="lambda_analysis",
                format="json"
            )

            logger.info(f"[Lambda Tracker] Logged {len(lambdas)} lambda values")
            logger.info(f"[Lambda Tracker] Mean={lambda_stats['lambda_mean']:.4f}, Std={lambda_stats['lambda_std']:.4f}")

        except Exception as e:
            logger.error(f"[Lambda Tracker] Failed: {e}")

# Context manager for automatic run management
class MLflowRunContext:
    """Context manager for MLflow runs"""

    def __init__(self, tracker: ExperimentTracker, run_name: str = None, **kwargs):
        self.tracker = tracker
        self.run_name = run_name
        self.kwargs = kwargs
        self.run_id = None

    def __enter__(self):
        self.run_id = self.tracker.start_run(self.run_name, **self.kwargs)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_run()
        self.tracker.cleanup_temp_artifacts()

# Convenience function
def create_experiment_tracker(experiment_name: str = None) -> ExperimentTracker:
    """Create experiment tracker with default settings"""
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return ExperimentTracker(experiment_name)

if __name__ == "__main__":
    # Example usage
    tracker = create_experiment_tracker("test_experiment")

    with MLflowRunContext(tracker, "test_run") as tracker:
        # Log some test data
        tracker.log_config({"test_param": 1.0, "nested": {"value": 2}})
        tracker.log_metrics({"test_metric": 0.95}, step=1)

        print("Experiment tracking test completed!")
