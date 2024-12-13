import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple
from abc import ABC, abstractmethod
import cupy as cp


class Softmax:
    """Computes softmax."""

    def __init__(self) -> None:
        """Initialize."""
        self.cache = {}

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Forward propagation.

        Args:
            x (cp.ndarray): Input.

        Returns:
            cp.ndarray: Softmax output.
        """

        # assert x.ndim == 2, "Input must be 2D."

        max = cp.max(x, axis=-1)[:, None]
        y = cp.exp(x - max) / cp.sum(cp.exp(x - max), axis=-1, keepdims=True)

        self.cache = dict(output=y)

        return y

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        Backward pass.

        Args:
            grad (cp.ndarray): Gradient with respect to the output.

        Returns:
            cp.ndarray: Gradient with respect to the input.
        """

        softmax = self.cache["output"]
        # ref - https://github.com/AkiRusProd/numpy-transformer/blob/master/transformer/activations.py

        J = softmax[..., cp.newaxis] * cp.tile(
            cp.identity(softmax.shape[-1]),
            (softmax.shape[0], *tuple(cp.ones(softmax.ndim, dtype=cp.int8).tolist())),
        ) - (
            softmax[..., cp.newaxis, :].transpose(
                *tuple(cp.arange(0, softmax.ndim - 1, 1, dtype=cp.int8).tolist()),
                -1,
                -2
            )
            @ softmax[..., cp.newaxis, :]
        )
        input_grad = grad[..., cp.newaxis, :] @ J
        return input_grad.reshape(grad.shape)

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)
