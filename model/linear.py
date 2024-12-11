import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple
from abc import ABC, abstractmethod
import cupy as cp
import copy

from model.optimizers import Optimizer, Adam


class Linear:
    """
    Linear layer with weight matrix of shape [out_features, in_features] for performance reasons.

    Performs the operation: z = x @ W.T + b (if bias is True)
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, init="he"
    ) -> None:
        """Initialize instance variables.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Include bias? Default is True.

        Returns:
            None
        """
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = cp.zeros(out_features)
        else:
            self.bias = None
        self.cache = dict(input=None)
        self._init_params(init=init)
        self.optimizer_w = None
        self.optimizer_b = None
        self._init_optimizers(optimizer=Adam())

    def _init_params(self, init="he") -> None:
        """Initialize weights and biases.

        Args:
            init (str): Initialization method. Default is "he". Options are "he", "xavier", "normal", "uniform".

        Returns:
            None
        """
        if init == "he":
            self.weight = cp.random.randn(
                self.out_features, self.in_features
            ) * cp.sqrt(2 / self.in_features)
        elif init == "xavier":
            self.weight = cp.random.randn(
                self.out_features, self.in_features
            ) * cp.sqrt(1 / self.in_features)
        elif init == "normal":
            self.weight = cp.random.randn(self.out_features, self.in_features)
        elif init == "uniform":
            self.weight = cp.random.uniform(
                -1, 1, (self.out_features, self.in_features)
            )
        else:
            raise ValueError("Invalid initialization method.")

        if self.bias is not None:
            self.bias = cp.zeros(self.out_features)

    def _init_optimizers(self, optimizer: Optimizer) -> None:
        """
        Initialize optimizers for weights and biases.

        Args:
            optimizer (Optimizer): Optimizer object.

        Returns:
            None
        """
        self.optimizer_w = copy.deepcopy(optimizer)
        self.optimizer_b = copy.deepcopy(optimizer)

    def _update_params(self) -> None:
        """
        Update weights and biases.

        Returns:
            None
        """
        self.weight = self.optimizer_w.update(self.grad_w, self.weight)
        if self.bias is not None:
            self.bias = self.optimizer_b.update(self.grad_b, self.bias)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass according to the operation: z = x @ W.T + b.

        Args:
            x (cp.ndarray): Input of shape [batch_size, in_features].

        Returns:
            cp.ndarray: Output of shape [batch_size, out_features].
        """

        # Cache input
        self.cache["input"] = x

        # Forward pass
        z = cp.dot(x, self.weight.T)
        if self.bias is not None:
            z += self.bias

        return z

    def backward(self, grad_z: cp.ndarray) -> cp.ndarray:
        """
        Backward pass for the linear layer.

        Args:
            grad_z (cp.ndarray): Gradient of the loss with respect to the output of the linear layer.

        Returns:
            cp.ndarray: Gradient of the loss with respect to the input of the linear layer.
        """
        # Compute gradients
        self.grad_w = cp.dot(grad_z.T, self.cache["input"])
        self.grad_b = cp.sum(grad_z, axis=0)
        grad_x = cp.dot(grad_z, self.weight)

        return grad_x

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def init_params(self, w: cp.ndarray, b: cp.ndarray) -> None:
        """
        Initialize weights and biases manually (for unit tests).

        Args:
            w (cp.ndarray): Weight matrix of shape [out_features, in_features].
            b (cp.ndarray): Bias vector of shape [out_features].

        Returns:
            None
        """
        if w.shape != (self.out_features, self.in_features):
            raise ValueError("Invalid shape for weight matrix.")
        if b.shape != (self.out_features,):
            raise ValueError("Invalid shape for bias vector.")

        self.weight = w
        self.bias = b

    def get_grads(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Get gradients of weights and biases.

        Returns:
            Tuple[cp.ndarray, cp.ndarray]: Gradients of weights and biases.
        """
        return self.grad_w, self.grad_b

    def get_weight(self) -> cp.ndarray:
        """
        Get weights.

        Returns:
            cp.ndarray: Weights.
        """
        return self.weight

    def get_bias(self) -> cp.ndarray:
        """
        Get biases.

        Returns:
            cp.ndarray: Biases.
        """
        return self.bias
