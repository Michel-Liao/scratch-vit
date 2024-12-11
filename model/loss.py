import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple
from abc import ABC, abstractmethod
import cupy as cp
import copy


class CategoricalCrossEntropyLoss:
    def __init__(self) -> None:
        pass

    def softmax(self, logits: cp.ndarray) -> cp.ndarray:
        """
        Applies softmax to logits along the class axis=1.

        Args:
            logits (cp.ndarray): Raw scores (B, C), where B is the batch size and C is the number of classes.

        Returns:
            cp.ndarray: Probabilities (B, C).
        """

        max_logit = cp.amax(logits, axis=1, keepdims=True)
        # Corrects softmax for numerical stability
        logits = logits - max_logit

        return cp.exp(logits) / cp.sum(cp.exp(logits), axis=1, keepdims=True)

    def forward(
        self, logits: cp.ndarray, labels: cp.ndarray
    ) -> Tuple[float, cp.ndarray]:
        """
        Applies softmax then cross-entropy loss over logits.

        Args:
            labels (cp.ndarray): One-hot ground truth labels (B, C).
            logits (cp.ndarray): Scores (B, C), where B is the batch size and C is the number of classes.

        Returns:
            Tuple[float, cp.ndarray]: Loss and probabilities (B, C).
        """
        # Convert logits to probabilities using softmax
        probs = cp.clip(self.softmax(logits), 1e-12, 1.0)  # Clip to avoid log(0)

        # Compute mean loss for mini-batch
        loss = -cp.sum(labels * cp.log(probs)) / logits.shape[0]

        return loss, probs

    def backward(self, probs, labels):
        """
        Computes gradient of cross-entropy loss with respect to logits.

        Args:
            probs (cp.ndarray): Softmax probabilities (B, C).
            labels (cp.ndarray): One-hot ground truth labels (B, C).

        Returns:
            cp.ndarray: Gradient of loss with respect to logits (B, C).
        """
        grad = (probs - labels) / labels.shape[0]

        return grad