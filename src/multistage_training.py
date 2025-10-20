"""
Multi-stage Training for Machine Precision PINNs
Based on DeepMind paper "Discovery of Unstable Singularities" (pages 17-18)

Key Principle (Eq. 19):
- Stage 1: Train coarse solution Φ̂_stage-1 (residual ~10^-8)
- Stage 2: Train error correction Φ̂_stage-2 to fix high-frequency residuals
- Final: Φ̂_exact = Φ̂_stage-1 + ε·Φ̂_stage-2 (residual ~10^-13)

The linearized equation for stage 2:
  -ε·D_k[Φ̂_stage-1]Φ̂_stage-2 + O(ε²) = R^stage-1_k

Where D_k is the linearized operator and R^stage-1_k is stage 1 residual.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import inspect
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from scipy.fft import fft2, fftfreq
import matplotlib.pyplot as plt

try:
    from .gauss_newton_optimizer_enhanced import MetaOptimizer
except ImportError:  # pragma: no cover - fallback for script execution
    from gauss_newton_optimizer_enhanced import MetaOptimizer  # type: ignore

try:
    import pytorch_lightning as pl  # type: ignore
    _LIGHTNING_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore
    _LIGHTNING_AVAILABLE = False

try:
    from accelerate import Accelerator  # type: ignore
    _ACCELERATE_AVAILABLE = True
except ImportError:
    Accelerator = None  # type: ignore
    _ACCELERATE_AVAILABLE = False

logger = logging.getLogger(__name__)


if _LIGHTNING_AVAILABLE:
    class PINNLightningModule(pl.LightningModule):
        """Minimal Lightning wrapper for PINN training."""

        def __init__(self,
                     network: nn.Module,
                     optimizer_cls: Callable = torch.optim.Adam,
                     lr: float = 1e-3):
            super().__init__()
            self.model = network
            self.optimizer_cls = optimizer_cls
            self.lr = lr

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def training_step(self, batch, batch_idx: int):
            inputs, targets = batch
            outputs = self(inputs)
            loss = torch.mean((outputs - targets) ** 2)
            self.log("train_loss", loss, on_step=True, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return self.optimizer_cls(self.model.parameters(), lr=self.lr)


else:
    class PINNLightningModule:  # type: ignore
        """Stub Lightning module when PyTorch Lightning is unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch Lightning is not installed; enable it to use Lightning features.")


@dataclass
class MultiStageConfig:
    """Configuration for multi-stage training"""
    # Stage 1 (coarse)
    stage1_epochs: int = 50000
    stage1_target_residual: float = 1e-8
    stage1_hidden_layers: List[int] = None  # [64, 64, 64]

    # Stage 2 (refinement)
    stage2_epochs: int = 100000
    stage2_target_residual: float = 1e-13
    stage2_use_fourier: bool = True
    stage2_fourier_sigma: Optional[float] = None  # Auto-compute from residual

    # General
    epsilon: float = 1.0  # Error scaling factor
    precision: torch.dtype = torch.float64
    device: str = "auto"
    verbose: bool = True
    checkpoint_frequency: int = 10000
    stage1_learning_rate: float = 1e-3
    meta_optimizer_lr: float = 1e-4
    use_fsdp: bool = False
    use_lightning: bool = False
    use_accelerate: bool = False
    optimizer: Optional[str] = None
    lightning_trainer_fn: Optional[Callable[..., Dict]] = None
    accelerate_training_fn: Optional[Callable[..., Dict]] = None

    def __post_init__(self):
        if self.stage1_hidden_layers is None:
            self.stage1_hidden_layers = [64, 64, 64, 64]

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class FourierFeatureNetwork(nn.Module):
    """
    Fourier Feature Network for Stage 2

    Paper methodology (page 18):
    - Stage 1 residual has high-frequency content
    - Use Fourier features: [cos(Bx), sin(Bx)]
    - B ~ N(0, σ²) where σ = 2π·f_d
    - f_d = dominant frequency from stage 1 residual

    This allows network to efficiently learn high-frequency corrections.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 fourier_sigma: float = 1.0,
                 num_fourier_features: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.fourier_sigma = fourier_sigma
        self.num_fourier_features = num_fourier_features

        # Fourier feature mapping: [cos(Bx), sin(Bx)]
        # B ~ N(0, σ²I) - sampled once and frozen
        self.register_buffer(
            'B',
            torch.randn(input_dim, num_fourier_features) * fourier_sigma
        )

        # MLP after Fourier features
        fourier_output_dim = 2 * num_fourier_features  # cos + sin
        self.mlp = nn.Sequential(
            nn.Linear(fourier_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

        logger.info(f"Fourier Feature Network initialized:")
        logger.info(f"  σ = {fourier_sigma:.6f}")
        logger.info(f"  Num features = {num_fourier_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Fourier features

        Args:
            x: Input coordinates [N, input_dim]

        Returns:
            Output [N, output_dim]
        """
        # Fourier feature mapping
        x_proj = x @ self.B  # [N, num_fourier_features]
        fourier_features = torch.cat([
            torch.cos(2 * np.pi * x_proj),
            torch.sin(2 * np.pi * x_proj)
        ], dim=-1)  # [N, 2*num_fourier_features]

        # MLP
        return self.mlp(fourier_features)


class MultiStageTrainer:
    """
    Multi-stage training orchestrator

    Implements 2-stage training pipeline:
    1. Stage 1: Coarse solution (standard PINN)
    2. Stage 2: High-frequency error correction (Fourier PINN)
    """

    def __init__(self, config: MultiStageConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.stage1_network = None
        self.stage2_network = None
        self.stage1_history = {}
        self.stage2_history = {}

        logger.info("Multi-stage Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Target precision: {config.stage2_target_residual:.2e}")

    def analyze_residual_frequency(self,
                                   residual: torch.Tensor,
                                   spatial_grid: torch.Tensor) -> float:
        """
        Analyze dominant frequency in stage 1 residual

        Paper: Use FFT to find f_d^(e), then set σ = 2π·f_d^(e)

        Args:
            residual: Stage 1 residual [N, M] or [N]
            spatial_grid: Spatial coordinates [N, M, dim]

        Returns:
            Dominant frequency f_d
        """
        # Convert to 2D array if needed
        if residual.dim() == 1:
            # Reshape based on spatial_grid
            if spatial_grid.dim() == 3:
                N, M, _ = spatial_grid.shape
                residual_2d = residual.reshape(N, M)
            else:
                # Assume square grid
                N = int(np.sqrt(len(residual)))
                residual_2d = residual[:N*N].reshape(N, N)
        else:
            residual_2d = residual

        # Convert to numpy for FFT
        residual_np = residual_2d.detach().cpu().numpy()

        # 2D FFT
        fft_result = fft2(residual_np)
        power_spectrum = np.abs(fft_result) ** 2

        # Get frequency grid
        N, M = residual_np.shape
        freq_y = fftfreq(N)
        freq_x = fftfreq(M)

        # Find dominant frequency (excluding DC component)
        power_spectrum[0, 0] = 0  # Remove DC
        max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)

        # Compute frequency magnitude
        fy = abs(freq_y[max_idx[0]])
        fx = abs(freq_x[max_idx[1]])
        dominant_freq = np.sqrt(fy**2 + fx**2)

        logger.info(f"Residual frequency analysis:")
        logger.info(f"  Dominant frequency f_d = {dominant_freq:.6f}")
        logger.info(f"  Fourier σ = 2π·f_d = {2*np.pi*dominant_freq:.6f}")

        return dominant_freq

    def train_stage1(self,
                    network: nn.Module,
                    train_function: Callable,
                    validation_function: Callable) -> Dict:
        """
        Train stage 1: Coarse solution

        Args:
            network: Stage 1 network (standard PINN)
            train_function: Training callback
            validation_function: Validation callback

        Returns:
            Training history
        """
        logger.info("="*60)
        logger.info("STAGE 1: Coarse Solution Training")
        logger.info("="*60)

        self.stage1_network = network.to(self.device)
        self._wrap_stage1_with_fsdp()

        optimizer = self._configure_stage1_optimizer(self.stage1_network)

        # Mixed Precision Training (Patch #3.1)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        use_amp = self.device.type == "cuda"
        if use_amp:
            logger.info("[AMP] Mixed precision training enabled for Stage 1")

        history = None

        if self.config.use_lightning:
            history = self._train_stage1_with_lightning(optimizer)
        elif self.config.use_accelerate:
            history = self._train_stage1_with_accelerate(
                train_function,
                optimizer,
                use_amp,
                scaler
            )

        if history is None:
            history = self._invoke_stage1_training(
                train_function,
                optimizer,
                use_amp,
                scaler
            )

        # Validate
        val_results = validation_function(self.stage1_network)
        final_residual = val_results['max_residual']

        logger.info(f"\nStage 1 Complete:")
        logger.info(f"  Final residual: {final_residual:.6e}")
        logger.info(f"  Target: {self.config.stage1_target_residual:.6e}")

        if final_residual > self.config.stage1_target_residual * 10:
            logger.warning(f"Stage 1 did not reach target precision!")

        self.stage1_history = {
            'training': history,
            'validation': val_results,
            'final_residual': final_residual
        }

        # Save checkpoint (Patch #1.2)
        ckpt_path = "checkpoint_stage1_final.pt"
        torch.save({
            "model_state_dict": self.stage1_network.state_dict(),
            "history": self.stage1_history,
            "config": self.config
        }, ckpt_path)
        logger.info(f"[Checkpoint] Stage 1 model saved at {ckpt_path}")

        return self.stage1_history

    def create_stage2_network(self,
                             input_dim: int,
                             output_dim: int,
                             stage1_residual: torch.Tensor,
                             spatial_grid: torch.Tensor) -> nn.Module:
        """
        Create stage 2 network with informed architecture

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            stage1_residual: Stage 1 residual for frequency analysis
            spatial_grid: Spatial grid

        Returns:
            Stage 2 network (Fourier feature network)
        """
        logger.info("\nCreating Stage 2 Network...")

        if self.config.stage2_use_fourier:
            # Analyze residual frequency (Patch #1.3 - Adaptive σ)
            if self.config.stage2_fourier_sigma is None:
                dominant_freq = self.analyze_residual_frequency(
                    stage1_residual, spatial_grid
                )
                fourier_sigma = 2 * np.pi * dominant_freq
                logger.info(f"[Adaptive σ] Using σ = {fourier_sigma:.4f} from residual analysis")
            else:
                fourier_sigma = self.config.stage2_fourier_sigma
                logger.info(f"[Manual σ] Using configured σ = {fourier_sigma:.4f}")

            # Create Fourier feature network
            network = FourierFeatureNetwork(
                input_dim=input_dim,
                hidden_dim=64,
                output_dim=output_dim,
                fourier_sigma=fourier_sigma,
                num_fourier_features=64
            )
        else:
            # Standard MLP (fallback)
            logger.info("Using standard MLP for stage 2")
            network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, output_dim)
            )

        return network.to(self.device)

    def train_stage2(self,
                    stage2_network: nn.Module,
                    train_function: Callable,
                    validation_function: Callable,
                    stage1_residual: torch.Tensor) -> Dict:
        """
        Train stage 2: Error correction

        The loss function should minimize:
          ε·D_k[Φ̂_stage-1]Φ̂_stage-2 - R^stage-1_k

        Args:
            stage2_network: Stage 2 network
            train_function: Training callback
            validation_function: Validation callback
            stage1_residual: Stage 1 residual (source term)

        Returns:
            Training history
        """
        logger.info("="*60)
        logger.info("STAGE 2: Error Correction Training")
        logger.info("="*60)

        self.stage2_network = stage2_network

        history = train_function(
            network=self.stage2_network,
            stage1_network=self.stage1_network,
            stage1_residual=stage1_residual,
            max_epochs=self.config.stage2_epochs,
            target_loss=self.config.stage2_target_residual,
            epsilon=self.config.epsilon,
            checkpoint_freq=self.config.checkpoint_frequency
        )

        # Save residual tracker plot (Patch #2.1)
        import matplotlib.pyplot as plt
        residual_history = history.get("loss", [])
        if residual_history:
            plt.figure(figsize=(10, 6))
            plt.semilogy(residual_history)
            plt.title("Stage 2 Residual Convergence")
            plt.xlabel("Epoch")
            plt.ylabel("Residual Loss")
            plt.grid(True, alpha=0.3)
            plt.savefig("stage2_residual_tracker.png", dpi=200, bbox_inches='tight')
            plt.close()
            logger.info("[Residual Tracker] Saved plot: stage2_residual_tracker.png")

        # Validate combined solution
        val_results = validation_function(
            self.stage1_network,
            self.stage2_network,
            epsilon=self.config.epsilon
        )
        final_residual = val_results['max_residual']

        logger.info(f"\nStage 2 Complete:")
        logger.info(f"  Final combined residual: {final_residual:.6e}")
        logger.info(f"  Target: {self.config.stage2_target_residual:.6e}")
        logger.info(f"  Improvement: {self.stage1_history['final_residual']/final_residual:.1f}x")

        if final_residual < 1e-12:
            logger.info("[+] MACHINE PRECISION ACHIEVED!")

        self.stage2_history = {
            'training': history,
            'validation': val_results,
            'final_residual': final_residual
        }

        return self.stage2_history

    def compose_solution(self,
                        x: torch.Tensor,
                        epsilon: Optional[float] = None) -> torch.Tensor:
        """
        Compose final solution: Φ̂_exact = Φ̂_stage-1 + ε·Φ̂_stage-2

        Args:
            x: Input coordinates
            epsilon: Error scaling (default: config.epsilon)

        Returns:
            Combined solution
        """
        if self.stage1_network is None or self.stage2_network is None:
            raise RuntimeError("Both stages must be trained first")

        if epsilon is None:
            epsilon = self.config.epsilon

        with torch.no_grad():
            u_stage1 = self.stage1_network(x)
            u_stage2 = self.stage2_network(x)
            u_combined = u_stage1 + epsilon * u_stage2

        return u_combined

    def plot_training_progress(self, save_path: Optional[str] = None):
        """
        Visualize multi-stage training progress

        Creates 3 plots:
        1. Stage 1 loss curve
        2. Stage 2 loss curve
        3. Residual comparison (stage 1 vs combined)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Stage 1 loss
        ax1 = axes[0]
        if 'loss_history' in self.stage1_history['training']:
            loss1 = self.stage1_history['training']['loss_history']
            ax1.semilogy(loss1, linewidth=2, color='blue')
            ax1.axhline(y=self.config.stage1_target_residual,
                       color='red', linestyle='--', label='Target')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Stage 1: Coarse Training', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Stage 2 loss
        ax2 = axes[1]
        if 'loss_history' in self.stage2_history['training']:
            loss2 = self.stage2_history['training']['loss_history']
            ax2.semilogy(loss2, linewidth=2, color='green')
            ax2.axhline(y=self.config.stage2_target_residual,
                       color='red', linestyle='--', label='Target')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('Stage 2: Refinement Training', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Residual comparison
        ax3 = axes[2]
        stages = ['Stage 1\n(Coarse)', 'Stage 2\n(Combined)']
        residuals = [
            self.stage1_history['final_residual'],
            self.stage2_history['final_residual']
        ]
        colors = ['blue', 'green']

        bars = ax3.bar(stages, residuals, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Max Residual', fontsize=12)
        ax3.set_title('Residual Improvement', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, residuals):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}',
                    ha='center', va='bottom', fontsize=10)

        # Add improvement factor
        improvement = residuals[0] / residuals[1]
        ax3.text(0.5, 0.95, f'Improvement: {improvement:.1f}×',
                transform=ax3.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")

        plt.show()

    @staticmethod
    def _callable_parameters(func: Callable) -> set:
        """Return parameter names accepted by a callable."""
        try:
            return set(inspect.signature(func).parameters.keys())
        except (TypeError, ValueError):
            return set()

    def _configure_stage1_optimizer(self, network: nn.Module):
        """Select and initialise the optimiser according to configuration."""
        opt_name = getattr(self.config, "optimizer", None)

        if opt_name == "meta":
            logger.info("[Optimizer] Using MetaOptimizer (Adam base).")
            return MetaOptimizer(
                network.parameters(),
                torch.optim.Adam,
                meta_lr=self.config.meta_optimizer_lr,
                lr=self.config.stage1_learning_rate
            )

        if opt_name == "kfac":
            logger.info("[Optimizer] Using K-FAC preconditioner (handled externally).")
            return None

        if opt_name not in (None, "meta", "kfac"):
            logger.warning(f"[Optimizer] Unknown optimizer '{opt_name}'. Falling back to default training function.")

        return None

    def _wrap_stage1_with_fsdp(self) -> None:
        """Optionally wrap the stage 1 network with FSDP."""
        if not getattr(self.config, "use_fsdp", False):
            return

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
        except ImportError:
            logger.error("[Distributed] FSDP requested but not available. Skipping distributed wrapping.")
            self.config.use_fsdp = False
            return

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            logger.warning("[Distributed] torch.distributed not initialised; cannot enable FSDP.")
            self.config.use_fsdp = False
            return

        self.stage1_network = FSDP(self.stage1_network)
        logger.info("[Distributed] Stage 1 network wrapped with FSDP.")

    def _train_stage1_with_lightning(self, optimizer) -> Optional[Dict[str, Any]]:
        """Delegate stage 1 training to a PyTorch Lightning trainer if configured."""
        if not self.config.use_lightning:
            return None

        if not _LIGHTNING_AVAILABLE:
            logger.error("[Lightning] PyTorch Lightning not installed; falling back to standard training.")
            return None

        if self.config.lightning_trainer_fn is None:
            logger.error("[Lightning] No `lightning_trainer_fn` provided; unable to run Lightning training.")
            return None

        try:
            return self.config.lightning_trainer_fn(self.stage1_network, optimizer, self.config)
        except Exception as exc:
            logger.error(f"[Lightning] Training callback failed: {exc}")
            return None

    def _train_stage1_with_accelerate(self,
                                      train_function: Callable,
                                      optimizer,
                                      use_amp: bool,
                                      scaler) -> Optional[Dict[str, Any]]:
        """Delegate stage 1 training to HuggingFace Accelerate if configured."""
        if not self.config.use_accelerate:
            return None

        if not _ACCELERATE_AVAILABLE:
            logger.error("[Accelerate] accelerate not installed; falling back to standard training.")
            return None

        if self.config.accelerate_training_fn is None:
            logger.error("[Accelerate] No `accelerate_training_fn` provided; unable to use accelerate.")
            return None

        accelerator = Accelerator()
        try:
            return self.config.accelerate_training_fn(
                self.stage1_network,
                accelerator,
                train_function,
                optimizer,
                self.config
            )
        except Exception as exc:
            logger.error(f"[Accelerate] Training callback failed: {exc}")
            return None

    def _invoke_stage1_training(self,
                                train_function: Callable,
                                optimizer,
                                use_amp: bool,
                                scaler) -> Dict[str, Any]:
        """Invoke the user-provided training function with optional arguments."""
        params = self._callable_parameters(train_function)
        train_kwargs = {
            "network": self.stage1_network,
            "max_epochs": self.config.stage1_epochs,
            "target_loss": self.config.stage1_target_residual,
            "checkpoint_freq": self.config.checkpoint_frequency
        }

        if "use_amp" in params:
            train_kwargs["use_amp"] = use_amp
        if "scaler" in params:
            train_kwargs["scaler"] = scaler
        if optimizer is not None and "optimizer" in params:
            train_kwargs["optimizer"] = optimizer
        elif self.config.optimizer == "kfac" and "optimizer" in params:
            train_kwargs["optimizer"] = "kfac"
        if "device" in params:
            train_kwargs["device"] = self.device

        history = train_function(**train_kwargs)

        if history is None:
            return {}

        if isinstance(history, dict):
            return history

        # Fallback: wrap non-dict outputs
        return {"loss_history": history}


def save_multistage_checkpoint(trainer: MultiStageTrainer,
                               path: str):
    """Save multi-stage training checkpoint"""
    checkpoint = {
        'config': trainer.config,
        'stage1_network': trainer.stage1_network.state_dict() if trainer.stage1_network else None,
        'stage2_network': trainer.stage2_network.state_dict() if trainer.stage2_network else None,
        'stage1_history': trainer.stage1_history,
        'stage2_history': trainer.stage2_history
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_multistage_checkpoint(path: str,
                               stage1_network: nn.Module,
                               stage2_network: Optional[nn.Module] = None) -> MultiStageTrainer:
    """Load multi-stage training checkpoint"""
    checkpoint = torch.load(path)

    trainer = MultiStageTrainer(checkpoint['config'])

    if checkpoint['stage1_network']:
        stage1_network.load_state_dict(checkpoint['stage1_network'])
        trainer.stage1_network = stage1_network

    if checkpoint['stage2_network'] and stage2_network:
        stage2_network.load_state_dict(checkpoint['stage2_network'])
        trainer.stage2_network = stage2_network

    trainer.stage1_history = checkpoint['stage1_history']
    trainer.stage2_history = checkpoint['stage2_history']

    logger.info(f"Checkpoint loaded from {path}")
    return trainer
