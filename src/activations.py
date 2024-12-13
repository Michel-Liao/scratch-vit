from abc import ABC, abstractmethod
import cupy as cp
import cupyx

class Activation(ABC):
    """Activation interface."""

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        """
        Call the forward method.

        Args:
            x (cp.ndarray): Input.

        Returns:
            cp.ndarray: Output.
        """
        return self.forward(x)

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

class GeLU(Activation):
    """
    The GeLU activation function calculated using the definition from the GeLU paper.
    """

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.x = x
        self.cdf = 0.5 * (1 + cupyx.scipy.special.erf(x / cp.sqrt(2)))
        return (x * self.cdf).astype(x.dtype)

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        pdf = cp.exp(-0.5 * cp.power(self.x, 2)) / cp.sqrt(2 * cp.pi)
        return (grad * (self.cdf + self.x * pdf)).astype(self.x.dtype)