import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple
from abc import ABC, abstractmethod
import cupy as cp
import copy

from model.optimizers import Optimizer, Adam


class Linear:
    """Linear layer."""

    def __init__(
        self, in_size: int, out_size: int, init_method: str = "he", bias: bool = True
    ) -> None:
        """
        Args:
            in_size (int): The number of input features.
            out_size (int): The number of output features.
            init_method (str): Initialization method for the weights.
            bias (bool): Boolean determining whether the layer will learn an additive bias.
        """

        self.in_size = in_size
        self.out_size = out_size

        self.w = cp.zeros([out_size, in_size])
        self.b = cp.zeros([out_size]) if bias else None

        self.grad_w = None
        self.grad_b = None

        self.cache = {"input": None}

        self._init_params(init_method)
        self.init_optimizer(Adam())

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def _init_params(self, method) -> None:
        """
        Initialize the layer weights and biases.

        Args:
            method (str): Initialization method. Options include "he", "xavier",
                          "normal", "uniform".

        Returns:
            None
        """

        if method == "he":
            self.w = cp.random.randn(self.out_size, self.in_size) * cp.sqrt(
                2 / self.in_size
            )
        elif method == "xavier":
            limit = cp.sqrt(6 / (self.in_size + self.out_size))
            self.w = cp.random.uniform(-limit, limit, (self.out_size, self.in_size))
        elif method == "normal":
            self.w = cp.random.randn(self.out_size, self.in_size)
        elif method == "uniform":
            self.w = cp.random.uniform(-1, -1, (self.out_size, self.in_size))
        else:
            raise ValueError(f"Invalid initialization method {method}")

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward propagation.

        Args:
            x (cp.ndarray): Input array.

        Returns:
            cp.ndarray: Computed linear layer output.
        """

        self.cache["input"] = x

        z = cp.dot(x, self.w.T)
        if self.b is not None:
            z += self.b

        return z

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        Backward propagation.

        Args:
            grad (cp.ndarray): Gradient of the loss with respect to the output of the linear layer.

        Returns:
            cp.ndarray: Gradient of the loss with respect to the input of the linear layer.
        """

        input = self.cache["input"]
        if input is None:
            raise ValueError("Input to linear layer is none.")

        output_grad = cp.dot(grad, self.w)
        if grad.ndim == 3:
            self.grad_w = cp.sum(cp.matmul(grad.transpose(0, 2, 1), input), axis=0)

            if self.b is not None:
                self.grad_b = cp.sum(grad, axis=(0, 1))
        elif grad.ndim == 2:
            self.grad_w = cp.dot(grad.T, input)
            if self.b is not None:
                self.grad_b = grad.sum(axis=0)
        else:
            raise ValueError(f"Invalid grad dimension of {grad.ndim}, expected 2 or 3.")

        return output_grad

    def init_optimizer(self, optimizer: Optimizer) -> None:
        """
        Initialize the layer optimizer.

        Args:
            optimizer (Optimizer): The layer optimizer.
        """

        self.optimizer_w = copy.deepcopy(optimizer)
        self.optimizer_b = copy.deepcopy(optimizer)

    def update_params(self) -> None:
        """
        Update weights based on the calculated gradients.
        """

        self.w = self.optimizer_w.update(self.grad_w, self.w)
        if self.b is not None:
            self.b = self.optimizer_b.update(self.grad_b, self.b)

    def init_params(self, w: cp.ndarray, b: cp.ndarray) -> None:
        """
        Initialize weights and biases manually (for unit tests).

        Args:
            w (cp.ndarray): Weight matrix of shape [out_features, in_features].
            b (cp.ndarray): Bias vector of shape [out_features].

        Returns:
            None
        """
        if w.shape != (self.out_size, self.in_size):
            raise ValueError("Invalid shape for weight matrix.")
        if b.shape[0] != (self.out_size):
            raise ValueError("Invalid shape for bias vector.")

        self.w = w
        self.b = b

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
        return self.w

    def get_bias(self) -> cp.ndarray:
        """
        Get biases.

        Returns:
            cp.ndarray: Biases.
        """
        return self.b
