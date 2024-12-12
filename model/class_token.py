import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import cupy as cp

from model.optimizers import Optimizer


class Parameter:
    """Parameter wrapper for CLS token."""

    def __init__(self, cls: cp.ndarray) -> None:
        """Initialize."""
        self.cls = cls
        self.optimizer = None

    def init_optimizer(self, optimizer: Optimizer) -> None:
        """Set optimizer.

        Args:
            optimizer (Optimizer): Optimizer
        """
        self.optimizer = copy.deepcopy(optimizer)

    def update_params(self) -> None:
        self.val = self.optimizer.update(self.cache["grad"], self.val)

    def backward(self, grad: cp.ndarray) -> None:
        """Backward propagation.

        Args:
            grad (cp.ndarray): The gradient with respect to the cls token

        Returns:
            None
        """
        self.cache = dict(grad=cp.sum(grad, axis=0)[None, :])
