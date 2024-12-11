from abc import ABC, abstractmethod
import cupy as cp
import cupyx
import torch


class Activation(ABC):
    """Activation interface."""

    @abstractmethod
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass.

        Args:
            x (cp.ndarray): Input.

        Returns:
            cp.ndarray: Output.
        """
        pass

    @abstractmethod
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        Backward pass.

        Args:
            grad (cp.ndarray): Gradient of the loss with respect to the output.

        Returns:
            cp.ndarray: Gradient of the loss with respect to the input.
        """
        pass


class ReLU(Activation):
    """
    ReLU activation function.
    """

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.x = x

        return cp.maximum(0, x)

    def backward(self, grad):
        x = self.x

        return grad * cp.where(x > 0, 1, 0).astype(x.dtype)


class FastGELU(Activation):
    """
    The GELU activation function calculated using its approximation from the GELU paper.
    """

    def forward(self, x: cp.ndarray) -> cp.ndarray:

        self.x = x

        return (
            0.5
            * x
            * (1 + cp.tanh(cp.sqrt(2 / cp.pi) * (x + 0.044715 * cp.power(x, 3))))
        ).astype(x.dtype)

    def backward(self, grad: cp.ndarray) -> cp.ndarray:

        x = self.x
        sech = lambda z: 2 / (cp.exp(z) + cp.exp(-z))

        return grad * (
            0.5 * cp.tanh(0.0356774 * cp.power(x, 3) + 0.797885 * x)
            + (0.0535161 * cp.power(x, 3) + 0.398942 * x)
            * cp.power(sech(0.0356774 * cp.power(x, 3) + 0.797885 * x), 2)
            + 0.5
        ).astype(x.dtype)


class GELU(Activation):
    """
    The GELU activation function calculated using the definition from the GELU paper.
    """

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.x = x
        self.cdf = 0.5 * (1 + cupyx.scipy.special.erf(x / cp.sqrt(2)))

        return (x * self.cdf).astype(x.dtype)

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        pdf = cp.exp(-0.5 * cp.power(self.x, 2)) / cp.sqrt(2 * cp.pi)

        return (grad * (self.cdf + self.x * pdf)).astype(self.x.dtype)
