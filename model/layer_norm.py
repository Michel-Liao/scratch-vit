import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple
from abc import ABC, abstractmethod
import cupy as cp
import cupyx
import copy

from model.optimizers import Optimizer, Adam


class LayerNorm:
    """
    Layer normalization.
    """

    def __init__(self, normalized_shape, eps=1e-5) -> None:
        """Initialize instance variables.

        Args:
            normalized_shape (int): Shape to normalize over.
            eps (float): Epsilon. Default is 1e-5.

        Returns:
            None
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = cp.ones(normalized_shape)
        self.bias = cp.zeros(normalized_shape)
        self.cache = {}

        self._init_optimizers(optimizer=Adam())

    def _init_optimizers(self, optimizer: Optimizer) -> None:
        """
        Initialize optimizers for weight and bias.

        Args:
            optimizer (Optimizer): Optimizer object.

        Returns:
            None
        """
        self.optimizer_w = copy.deepcopy(optimizer)
        self.optimizer_b = copy.deepcopy(optimizer)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass through LayerNorm.

        Args:
            x (cp.ndarray): Input tensor of shape (..., normalized_shape)

        Returns:
            cp.ndarray: Normalized output tensor
        """
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"Expected last dimension to be {self.normalized_shape}, got {x.shape[-1]}"
            )

        self.cache["input"] = x

        mean = cp.mean(x, axis=-1, keepdims=True)
        var = cp.var(x, axis=-1, keepdims=True)

        x_centered = x - mean

        rec_stddev = cupyx.rsqrt(var + self.eps)

        x_norm = x_centered * rec_stddev

        x_scaled = self.weight * x_norm + self.bias

        # Cache for backward pass
        self.cache.update(
            {
                "mean": mean,
                "var": var,
                "x_centered": x_centered,
                "rec_stddev": rec_stddev,
                "x_norm": x_norm,
            }
        )

        return x_scaled

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """Backward propagation through LayerNorm.

        Args:
            grad_output (cp.ndarray): Gradient of the loss with respect to the output.

        Returns:
            cp.ndarray: Gradient of the loss with respect to the input.
        """
        N = self.normalized_shape
        x_norm = self.cache["x_norm"]
        x_centered = self.cache["x_centered"]
        rec_stddev = self.cache["rec_stddev"]

        # Calculate gradients for weight and bias
        self.grad_w = cp.sum(
            grad_output * x_norm, axis=tuple(range(grad_output.ndim - 1))
        )
        self.grad_b = cp.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))

        # Gradient with respect to normalized input
        dx_norm = grad_output * self.weight

        # Gradient with respect to variance
        dvar = (
            cp.sum(dx_norm * x_centered, axis=-1, keepdims=True) * -0.5 * rec_stddev**3
        )

        # Gradient with respect to mean
        dmu1 = cp.sum(dx_norm * -rec_stddev, axis=-1, keepdims=True)
        dmu2 = dvar * -2.0 * cp.mean(x_centered, axis=-1, keepdims=True)
        dmu = dmu1 + dmu2

        # Gradient with respect to input
        dx1 = dx_norm * rec_stddev
        dx2 = dvar * 2.0 * x_centered / N
        dx3 = dmu / N

        return dx1 + dx2 + dx3

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def update_params(self) -> None:
        """
        Update weight and bias.

        Returns:
            None
        """
        self.weight = self.optimizer_w.update(self.grad_g, self.weight)
        if self.bias is not None:
            self.bias = self.optimizer_b.update(self.grad_b, self.bias)

    def get_grads(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Get gradients of weights and biases.

        Returns:
            Tuple[cp.ndarray, cp.ndarray]: Gradients of weights and biases.
        """
        return self.grad_w, self.grad_b
