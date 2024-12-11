import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import cupy as cp

from model.loss import CategoricalCrossEntropyLoss


class TestCategoricalCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        """
        Init loss functions and tolerances.
        """
        self.loss_fn = CategoricalCrossEntropyLoss()
        self.torch_loss_fn = nn.CrossEntropyLoss()
        # Tolerances
        self.rtol = 1e-5
        self.atol = 1e-8

    def test_manual(self):
        """
        Runs simple 2D input against manually calculated expected output.
        """
        logits = cp.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        expected = cp.array(
            [[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]]
        )
        output = self.loss_fn.softmax(logits)
        cp.testing.assert_allclose(output, expected, rtol=self.rtol, atol=self.atol)

    def test_forward(self):
        """
        Test forward pass against PyTorch implementation
        """

        logits = cp.array([[2.0, 1.0, 0.16], [0.07, 2.0, 3.0]])
        labels = cp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Custom implementation
        loss, probs = self.loss_fn.forward(logits, labels)

        # PyTorch implementation
        torch_logits = torch.tensor(cp.asnumpy(logits), dtype=torch.float32)
        torch_labels = torch.tensor(cp.asnumpy(labels).argmax(axis=1))
        torch_loss = self.torch_loss_fn(torch_logits, torch_labels)

        # Compare losses
        np.testing.assert_allclose(
            loss.get(), torch_loss.item(), rtol=self.rtol, atol=self.atol
        )

        # Check probabilities sum to 1
        cp.testing.assert_allclose(
            cp.sum(probs, axis=1), cp.ones(probs.shape[0]), rtol=self.rtol
        )

    def test_backward(self):
        """
        Test backward pass gradients
        """

        logits = cp.array([[2.0, 1.0, 0.16], [0.07, 2.0, 3.0]])
        labels = cp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Forward pass
        _, probs = self.loss_fn.forward(logits, labels)

        # Backward pass
        grad = self.loss_fn.backward(probs, labels)

        # PyTorch backward pass
        torch_logits = torch.tensor(
            cp.asnumpy(logits), dtype=torch.float32, requires_grad=True
        )
        torch_labels = torch.tensor(cp.asnumpy(labels).argmax(axis=1))
        torch_loss = self.torch_loss_fn(torch_logits, torch_labels)
        torch_loss.backward()

        # Compare gradients
        np.testing.assert_allclose(
            cp.asnumpy(grad), torch_logits.grad.numpy(), rtol=self.rtol, atol=self.atol
        )

    def test_numerical_stability(self):
        """
        Runs forward pass with small and large logits to check for numerical stability.
        """
        # Small logits
        logits = cp.array([[-1e-7, -1e-7, -1e-7]])
        labels = cp.array([[1.0, 0.0, 0.0]])
        loss, probs = self.loss_fn.forward(logits, labels)
        self.assertTrue(cp.isfinite(loss))
        self.assertTrue(cp.all(cp.isfinite(probs)))

        # Large logits
        logits = cp.array([[1e7, 1e7, 1e7]])
        loss, probs = self.loss_fn.forward(logits, labels)
        self.assertTrue(cp.isfinite(loss))
        self.assertTrue(cp.all(cp.isfinite(probs)))


if __name__ == "__main__":
    unittest.main()
