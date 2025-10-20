"""
High-Precision Gauss-Newton Optimizer for PINNs
Based on DeepMind "Discovery of Unstable Singularities" (arXiv:2509.14185)

Enhanced Features (DeepMind Paper):
- Rank-1 unbiased Hessian estimator for memory efficiency
- Exponential moving average (EMA) for Hessian approximation
- Automated learning rate computation
- Second-order optimization for near machine precision
- Adaptive Levenberg-Marquardt damping
- Line search for robust convergence
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GaussNewtonConfig:
    """Configuration for enhanced Gauss-Newton optimizer"""
    # Core optimization
    learning_rate: float = 1e-3
    damping_factor: float = 1e-6  # Levenberg-Marquardt damping
    max_iterations: int = 1000
    tolerance: float = 1e-12  # Near machine precision target

    # Line search
    line_search: bool = True
    line_search_max_iter: int = 20
    line_search_c1: float = 1e-4  # Armijo condition

    # Adaptive damping
    adaptive_damping: bool = True
    damping_increase: float = 10.0
    damping_decrease: float = 0.1

    # Gradient clipping (increased for ill-conditioned problems)
    gradient_clip: float = 10.0

    # Krylov solver (optional)
    use_krylov_solver: bool = False
    krylov_max_iter: int = 200
    krylov_tol: float = 1e-10

    # EMA for Hessian approximation (DeepMind enhancement)
    use_ema_hessian: bool = True
    ema_decay: float = 0.9  # β for exponential moving average

    # Rank-1 Hessian estimator (DeepMind enhancement)
    use_rank1_hessian: bool = True
    rank1_batch_size: int = 10  # Number of samples for rank-1 approximation

    # Automated learning rate
    auto_learning_rate: bool = True
    lr_update_freq: int = 10  # Update learning rate every N iterations

    # Early stopping (Patch #1.1)
    early_stop_threshold: Optional[float] = None  # Stop when loss < threshold

    # Trust-Region Damping (Patch #4.2)
    trust_radius: Optional[float] = None  # Adaptive damping via trust-region

    # Precision and verbosity
    precision: torch.dtype = torch.float64
    verbose: bool = True


class MetaOptimizer:
    """Light-weight meta-optimizer that wraps a base optimizer and adjusts its LR."""

    def __init__(self,
                 params,
                 base_optimizer_cls,
                 meta_lr: float = 1e-3,
                 lr: float = 1e-3,
                 **kwargs):
        self.meta_lr = meta_lr
        self.base_optimizer = base_optimizer_cls(params, lr=lr, **kwargs)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def step(self, closure=None):
        result = self.base_optimizer.step(closure)
        noise = 1.0 + self.meta_lr * torch.randn(1, dtype=torch.float64).item()
        for group in self.base_optimizer.param_groups:
            group["lr"] = max(group.get("lr", 1e-3) * noise, 1e-12)
        return result


class Rank1HessianEstimator:
    """
    Rank-1 unbiased Hessian estimator for memory-efficient second-order optimization

    Based on:
    - Martens & Grosse (2015): "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
    - Schraudolph (2002): "Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent"

    Key Idea:
    Instead of computing full Hessian H (O(P²) memory), use rank-1 approximation:
        H ≈ E[∇r · ∇r^T]
    where ∇r is gradient of individual residual
    """

    def __init__(self, n_params: int, batch_size: int = 10, dtype=torch.float64):
        self.n_params = n_params
        self.batch_size = batch_size
        self.dtype = dtype

        logger.info(f"Rank-1 Hessian Estimator initialized: {n_params} params, batch_size={batch_size}")

    def estimate_hessian_vector_product(self,
                                       jacobian: torch.Tensor,
                                       vector: torch.Tensor,
                                       residual_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Compute Hessian-vector product Hv using rank-1 approximation

        H·v ≈ J^T J·v = J^T (J·v)

        For memory efficiency, use batch sampling:
            H·v ≈ (1/B) Σ_i (J_i^T J_i)·v

        Args:
            jacobian: Full Jacobian [N, P] or sampled
            vector: Direction vector [P]
            residual_indices: Optional indices for sampling

        Returns:
            Hv: Hessian-vector product [P]
        """
        if residual_indices is None:
            # Random sampling
            n_residuals = jacobian.shape[0]
            indices = torch.randperm(n_residuals)[:self.batch_size]
        else:
            indices = torch.tensor(residual_indices)

        # Sample Jacobian rows
        J_sample = jacobian[indices]  # [B, P]

        # Compute Jv
        Jv = torch.matmul(J_sample, vector)  # [B]

        # Compute J^T(Jv)
        Hv = torch.matmul(J_sample.T, Jv)  # [P]

        # Scale by batch size for unbiased estimate
        Hv = Hv * (jacobian.shape[0] / self.batch_size)

        return Hv


class EMAHessianApproximation:
    """
    Exponential Moving Average for Hessian approximation

    H_t = β·H_{t-1} + (1-β)·(J^T J)_t

    This provides smoothed second-order information and reduces
    noise in Hessian estimation.
    """

    def __init__(self, n_params: int, decay: float = 0.9, dtype=torch.float64):
        self.n_params = n_params
        self.decay = decay
        self.dtype = dtype

        # EMA state (stored as diagonal approximation for memory efficiency)
        self.ema_diag = torch.ones(n_params, dtype=dtype)
        self.initialized = False

        logger.info(f"EMA Hessian initialized: n_params={n_params}, decay={decay}")

    def update(self, jacobian: torch.Tensor):
        """
        Update EMA with new Jacobian

        For memory efficiency, only track diagonal:
            H_diag ≈ diag(J^T J) = Σ_i J_{i,j}²

        Args:
            jacobian: Jacobian matrix [N, P]
        """
        # Compute diagonal of J^T J
        jtj_diag = torch.sum(jacobian ** 2, dim=0)  # [P]

        if not self.initialized:
            # Initialize with correct device
            self.ema_diag = jtj_diag.clone()
            self.initialized = True
        else:
            # EMA update (ensure same device)
            self.ema_diag = self.ema_diag.to(jacobian.device)
            self.ema_diag = self.decay * self.ema_diag + (1 - self.decay) * jtj_diag

    def get_preconditioner(self, damping: float = 1e-6) -> torch.Tensor:
        """
        Get diagonal preconditioner from EMA Hessian

        Returns:
            Diagonal preconditioner [P] for preconditioning: (H + λI)^{-1}
        """
        return 1.0 / (self.ema_diag + damping)


class HighPrecisionGaussNewtonEnhanced:
    """
    Enhanced High-precision Gauss-Newton optimizer

    Enhancements over standard implementation:
    1. Rank-1 Hessian estimator for memory efficiency
    2. EMA for smoothed second-order information
    3. Automated learning rate based on curvature
    4. Adaptive Levenberg-Marquardt damping
    5. Robust line search
    """

    def __init__(self, config: GaussNewtonConfig):
        self.config = config

        # Optimization state
        self.current_parameters = None
        self.current_loss = None

        # History tracking
        self.loss_history = []
        self.gradient_norm_history = []
        self.damping_history = []
        self.step_size_history = []
        self.lr_history = []

        # Adaptive damping state
        self.damping = config.damping_factor
        self.learning_rate = config.learning_rate

        # Enhanced components
        self.rank1_estimator = None
        self.ema_hessian = None

        logger.info("Enhanced High-Precision Gauss-Newton Optimizer initialized")
        logger.info(f"  Target tolerance: {config.tolerance:.2e}")
        logger.info(f"  Rank-1 Hessian: {config.use_rank1_hessian}")
        logger.info(f"  EMA Hessian: {config.use_ema_hessian}")
        logger.info(f"  Auto LR: {config.auto_learning_rate}")

    def _initialize_enhanced_components(self, n_params: int):
        """Initialize rank-1 estimator and EMA Hessian"""
        if self.config.use_rank1_hessian and self.rank1_estimator is None:
            self.rank1_estimator = Rank1HessianEstimator(
                n_params=n_params,
                batch_size=self.config.rank1_batch_size,
                dtype=self.config.precision
            )

        if self.config.use_ema_hessian and self.ema_hessian is None:
            self.ema_hessian = EMAHessianApproximation(
                n_params=n_params,
                decay=self.config.ema_decay,
                dtype=self.config.precision
            )

    def _compute_loss(self, residual: torch.Tensor) -> float:
        """Compute loss as 0.5 * ||r||²"""
        return 0.5 * torch.sum(residual**2).item()

    def _compute_gradient(self, residual: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Compute gradient g = J^T r"""
        return torch.matmul(jacobian.T, residual)

    def update_damping(self, loss_reduction_ratio: float):
        """
        Adaptive damping via trust-region style update (Patch #4.2)

        Args:
            loss_reduction_ratio: Actual loss reduction / predicted loss reduction
        """
        if self.config.trust_radius is None:
            return

        if loss_reduction_ratio < 0.25:
            # Poor step, increase damping
            self.damping *= 2.0
            logger.info(f"[Trust-Region] Poor step (ratio={loss_reduction_ratio:.3f}), increasing damping")
        elif loss_reduction_ratio > 0.75:
            # Good step, decrease damping
            self.damping *= 0.5
            logger.info(f"[Trust-Region] Good step (ratio={loss_reduction_ratio:.3f}), decreasing damping")

        # Clamp damping
        self.damping = max(self.damping, 1e-12)
        logger.info(f"[Trust-Region] Updated damping to {self.damping:.3e}")

    def _solve_gauss_newton_system(self, jacobian: torch.Tensor,
                                   residual: torch.Tensor) -> torch.Tensor:
        """
        Solve Gauss-Newton system with enhancements:
        (J^T J + λI) δ = -J^T r

        With EMA preconditioning:
        P(J^T J + λI) δ = -P J^T r
        where P is diagonal preconditioner from EMA Hessian
        """
        n_params = jacobian.shape[1]

        # Update EMA Hessian
        if self.config.use_ema_hessian and self.ema_hessian is not None:
            self.ema_hessian.update(jacobian)

        # Compute J^T J and J^T r
        JTJ = torch.matmul(jacobian.T, jacobian)
        JTr = torch.matmul(jacobian.T, residual)

        # Add Levenberg-Marquardt damping (with device consistency)
        damped_JTJ = JTJ + self.damping * torch.eye(
            n_params,
            dtype=self.config.precision,
            device=jacobian.device
        )

        # Apply EMA preconditioning if available
        if self.config.use_ema_hessian and self.ema_hessian is not None:
            precond = self.ema_hessian.get_preconditioner(self.damping)
            damped_JTJ = damped_JTJ * precond.unsqueeze(0)
            JTr = JTr * precond

        rhs = -JTr

        # Optional Krylov solver for large systems
        if self.config.use_krylov_solver:
            try:
                cg_kwargs = {"maxiter": self.config.krylov_max_iter}
                try:
                    step, cg_info = torch.linalg.cg(
                        damped_JTJ,
                        rhs,
                        atol=self.config.krylov_tol,
                        **cg_kwargs
                    )
                except TypeError:
                    # Older PyTorch versions may not support 'atol'; retry without it
                    step, cg_info = torch.linalg.cg(
                        damped_JTJ,
                        rhs,
                        **cg_kwargs
                    )

                if cg_info == 0:
                    logger.info("[Gauss-Newton] Solved linear system via Krylov CG")
                    return step

                logger.warning(f"[Gauss-Newton] CG solver returned info={cg_info}; falling back to direct solve")
            except (RuntimeError, AttributeError) as err:
                logger.warning(f"[Gauss-Newton] Krylov solver unavailable ({err}); falling back to direct solve")

        try:
            # Solve linear system
            step = torch.linalg.solve(damped_JTJ, rhs)
            return step
        except RuntimeError as e:
            logger.warning(f"Linear system solve failed: {e}. Using gradient descent fallback.")
            gradient = self._compute_gradient(residual, jacobian)
            return -self.learning_rate * gradient

    def _update_learning_rate(self, jacobian: torch.Tensor, iteration: int):
        """
        Automatically update learning rate based on curvature

        Heuristic: lr ∝ 1/||J^T J||
        """
        if not self.config.auto_learning_rate:
            return

        if iteration % self.config.lr_update_freq != 0:
            return

        # Estimate largest eigenvalue of J^T J (power method approximation)
        # λ_max ≈ ||J||²
        spectral_norm = torch.linalg.matrix_norm(jacobian, ord=2).item()
        curvature = spectral_norm ** 2

        # Update learning rate
        if curvature > 1e-12:
            new_lr = min(1.0 / curvature, 1.0)
            self.learning_rate = 0.9 * self.learning_rate + 0.1 * new_lr  # Smooth update
            self.lr_history.append(self.learning_rate)

            if self.config.verbose and iteration % 100 == 0:
                logger.info(f"  Auto LR updated: {self.learning_rate:.2e} (curvature={curvature:.2e})")

    def step_with_kfac(self,
                       loss_fn: Callable,
                       dataloader,
                       model: nn.Module,
                       lr: float = 1e-3,
                       factor_decay: float = 0.95) -> None:
        """
        Perform a single K-FAC preconditioned optimisation step.

        Args:
            loss_fn: Callable taking (predictions, targets) -> loss tensor.
            dataloader: Iterable yielding batches of (inputs, targets).
            model: Model to update using K-FAC.
            lr: K-FAC optimiser learning rate.
            factor_decay: Decay rate for Kronecker factors.
        """
        try:
            import kfac  # type: ignore
        except ImportError:
            logger.warning("[Optimizer] K-FAC not installed. Skipping K-FAC update.")
            return

        optimizer = kfac.KFACOptimizer(model, lr=lr, factor_decay=factor_decay)
        model.train()

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                logger.warning("[Optimizer] Unexpected batch format for K-FAC step; skipping batch.")
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        logger.info("[Optimizer] Completed K-FAC preconditioned step.")

    def _line_search(self, compute_residual_fn: Callable,
                    parameters: torch.Tensor,
                    step: torch.Tensor,
                    current_loss: float,
                    gradient: torch.Tensor) -> Tuple[float, bool]:
        """
        Armijo line search with backtracking
        """
        if not self.config.line_search:
            return 1.0, True

        alpha = 1.0
        c1 = self.config.line_search_c1

        directional_derivative = torch.dot(gradient, step).item()

        for i in range(self.config.line_search_max_iter):
            test_params = parameters + alpha * step
            test_residual = compute_residual_fn(test_params)
            test_loss = self._compute_loss(test_residual)

            # Armijo condition
            if test_loss <= current_loss + c1 * alpha * directional_derivative:
                return alpha, True

            alpha *= 0.5

        logger.warning("Line search failed")
        return 0.01, False

    def _update_damping(self, loss_reduction: float, predicted_reduction: float):
        """Update Levenberg-Marquardt damping based on step quality"""
        if not self.config.adaptive_damping:
            return

        if predicted_reduction > 0:
            ratio = loss_reduction / predicted_reduction

            if ratio > 0.75:
                self.damping *= self.config.damping_decrease
            elif ratio < 0.25:
                self.damping *= self.config.damping_increase
        else:
            self.damping *= self.config.damping_increase

        self.damping = torch.clamp(torch.tensor(self.damping), 1e-12, 1e6).item()

    def optimize(self,
                compute_residual_fn: Callable,
                compute_jacobian_fn: Callable,
                initial_parameters: torch.Tensor) -> Dict:
        """
        Main optimization loop

        Args:
            compute_residual_fn: Function that computes residual vector r(θ)
            compute_jacobian_fn: Function that computes Jacobian J = ∂r/∂θ
            initial_parameters: Starting parameters [P]

        Returns:
            Optimization results dictionary
        """
        logger.info("Starting enhanced Gauss-Newton optimization...")

        self.current_parameters = initial_parameters.clone().detach()
        n_params = len(self.current_parameters)

        # Initialize enhanced components
        self._initialize_enhanced_components(n_params)

        # Reset history
        self.loss_history = []
        self.gradient_norm_history = []
        self.damping_history = []
        self.step_size_history = []
        self.lr_history = []

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            # Compute residual and Jacobian
            current_residual = compute_residual_fn(self.current_parameters)
            current_jacobian = compute_jacobian_fn(self.current_parameters)
            self.current_loss = self._compute_loss(current_residual)

            # Compute gradient
            gradient = self._compute_gradient(current_residual, current_jacobian)
            gradient_norm = torch.norm(gradient).item()

            # Record history
            self.loss_history.append(self.current_loss)
            self.gradient_norm_history.append(gradient_norm)
            self.damping_history.append(self.damping)

            # Check convergence
            if self.current_loss < self.config.tolerance:
                logger.info(f"Converged! Loss: {self.current_loss:.2e} < {self.config.tolerance:.2e}")
                break

            # Early stopping check (Patch #1.1)
            if self.config.early_stop_threshold is not None:
                if self.current_loss < self.config.early_stop_threshold:
                    logger.info(f"[Early Stop] Iteration {iteration}: Loss {self.current_loss:.3e} < threshold {self.config.early_stop_threshold:.3e}")
                    break

            if gradient_norm < self.config.tolerance:
                logger.info(f"Gradient convergence! ||g||: {gradient_norm:.2e}")
                break

            # Update learning rate
            self._update_learning_rate(current_jacobian, iteration)

            # Compute Gauss-Newton step
            step = self._solve_gauss_newton_system(current_jacobian, current_residual)

            # Clip step for stability
            step_norm = torch.norm(step).item()
            if step_norm > self.config.gradient_clip:
                step = step * (self.config.gradient_clip / step_norm)
                step_norm = self.config.gradient_clip

            # Line search
            step_size, _ = self._line_search(
                compute_residual_fn,
                self.current_parameters,
                step,
                self.current_loss,
                gradient
            )
            self.step_size_history.append(step_size)

            # Predicted reduction
            predicted_reduction = -torch.dot(gradient, step).item()

            # Update parameters
            old_loss = self.current_loss
            self.current_parameters = self.current_parameters + step_size * step

            # Compute new loss
            new_residual = compute_residual_fn(self.current_parameters)
            new_loss = self._compute_loss(new_residual)
            loss_reduction = old_loss - new_loss

            # Update damping
            self._update_damping(loss_reduction, predicted_reduction)

            # Progress logging
            if self.config.verbose and (iteration % 100 == 0 or iteration < 10):
                logger.info(f"Iter {iteration:4d}: Loss={self.current_loss:.2e}, "
                          f"||g||={gradient_norm:.2e}, λ={self.damping:.2e}, "
                          f"α={step_size:.3f}, lr={self.learning_rate:.2e}")

        end_time = time.time()

        # Final results
        final_residual = compute_residual_fn(self.current_parameters)
        final_loss = self._compute_loss(final_residual)
        final_jacobian = compute_jacobian_fn(self.current_parameters)
        final_gradient = self._compute_gradient(final_residual, final_jacobian)
        final_gradient_norm = torch.norm(final_gradient).item()

        results = {
            'parameters': self.current_parameters,
            'loss': final_loss,
            'gradient_norm': final_gradient_norm,
            'iterations': iteration + 1,
            'converged': final_loss < self.config.tolerance or final_gradient_norm < self.config.tolerance,
            'optimization_time': end_time - start_time,
            'loss_history': self.loss_history,
            'gradient_norm_history': self.gradient_norm_history,
            'damping_history': self.damping_history,
            'step_size_history': self.step_size_history,
            'lr_history': self.lr_history
        }

        logger.info(f"[+] Optimization completed in {results['optimization_time']:.2f}s")
        logger.info(f"[+] Final loss: {final_loss:.2e}")
        logger.info(f"[+] Final gradient norm: {final_gradient_norm:.2e}")
        logger.info(f"[+] Converged: {results['converged']}")

        return results


# Test function
if __name__ == "__main__":
    print("[*] Testing Enhanced Gauss-Newton Optimizer")
    print("=" * 60)

    # Simple quadratic test problem
    n_params = 10
    n_residuals = 20

    # Ground truth
    true_params = torch.randn(n_params, dtype=torch.float64)
    A = torch.randn(n_residuals, n_params, dtype=torch.float64)
    b = torch.matmul(A, true_params)

    # Residual and Jacobian functions
    def compute_residual(params):
        return torch.matmul(A, params) - b

    def compute_jacobian(params):
        return A

    # Configure optimizer
    config = GaussNewtonConfig(
        learning_rate=1e-2,
        max_iterations=100,
        tolerance=1e-12,
        use_ema_hessian=True,
        use_rank1_hessian=True,
        auto_learning_rate=True,
        verbose=True
    )

    optimizer = HighPrecisionGaussNewtonEnhanced(config)

    # Initial guess
    initial = torch.randn(n_params, dtype=torch.float64)

    print(f"Initial loss: {0.5 * torch.sum(compute_residual(initial)**2).item():.2e}")

    # Optimize
    results = optimizer.optimize(compute_residual, compute_jacobian, initial)

    print(f"\n[+] Results:")
    print(f"  Final loss: {results['loss']:.2e}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Converged: {results['converged']}")
    print(f"  Parameter error: {torch.norm(results['parameters'] - true_params).item():.2e}")
    print("\n[+] Test complete!")
