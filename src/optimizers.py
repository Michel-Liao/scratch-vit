from abc import ABC, abstractmethod
import cupy as cp

class Optimizer(ABC):
    """
    Optimizer interface.
    """

    @abstractmethod
    def update(self, grad: cp.ndarray, w: cp.ndarray) -> cp.ndarray:
        """
        Update using gradient descent.

        Args:
            grad (cp.ndarray): Gradient of the loss with respect to the weights.
            w (cp.ndarray): Weights to be updated.

        Returns:
            cp.ndarray: Updated weights.
        """
        pass

class Adam(Optimizer):
    """
    The Adam optimizer as implemented in Algorithm 1 of the paper.
    """

    def __init__(
        self, lr: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
    ) -> None:
        """
        Args:
            lr (float): Learning rate.
            b1 (float): Exponential decay rate for the first moment estimates.
            b2 (float): Exponential decay rate for the second moment estimates.
            eps (float): A small constant for numerical stability.
        """
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, grad: cp.ndarray, w: cp.ndarray) -> cp.ndarray:
        """
        Perform one Adam optimization step.

        Args:
            grad (cp.ndarray): Gradient of the loss with respect to the weights.
            w (cp.ndarray): Current weights.

        Returns:
            cp.ndarray: Updated parameter values.
        """
        # Gradient check
        if cp.any(cp.isnan(grad)) or cp.any(cp.isinf(grad)):
            raise ValueError(
                f"Gradient contains NaN {cp.any(cp.isnan(grad))} or infinite {cp.any(cp.isinf(grad))} values"
            )

        # Initialize momentum vectors
        if self.m is None or self.v is None:
            self.m = cp.zeros_like(grad)
            self.v = cp.zeros_like(grad)

        # Timestep
        self.t += 1

        # Update biased first and second raw moment estimate
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)

        # Compute bias-corrected first and second moment estimates
        corr_m = self.m / (1 - self.b1**self.t)
        corr_v = self.v / (1 - self.b2**self.t)

        # Compute update
        update = (self.lr * corr_m) / (cp.sqrt(corr_v) + self.eps)

        return w - update
