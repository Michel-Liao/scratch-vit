import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import cupy as cp

from model.optimizers import Adam


class TestAdamOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        """
        Init with same random seeds and optimizer parameters for reproducibility.
        """
        np.random.seed(16)
        torch.manual_seed(16)
        cp.random.seed(16)

        self.lr = 0.001
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8

        # Initialize custom Adam optimizer
        self.adam = Adam(lr=self.lr, b1=self.b1, b2=self.b2, eps=self.eps)

    def test_single_update_step(self):
        """
        Test a single optimization step on a quadratic function.
        """
        # Create identical parameters for both optimizers
        w_custom = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        w_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Create gradient
        grad_custom = cp.array([0.1, 0.2, 0.3], dtype=cp.float32)
        grad_torch = torch.tensor([0.1, 0.2, 0.3])

        # Initialize PyTorch Adam optimizer
        torch_adam = torch.optim.Adam(
            [w_torch], lr=self.lr, betas=(self.b1, self.b2), eps=self.eps
        )

        # Update with custom Adam
        w_custom_updated = self.adam.update(grad_custom, w_custom)

        # Update with PyTorch Adam
        torch_adam.zero_grad()
        w_torch.grad = grad_torch
        torch_adam.step()

        # Compare
        np.testing.assert_allclose(
            cp.asnumpy(w_custom_updated),
            w_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Single update step produces different results",
        )

    def test_multiple_update_steps(self):
        """
        Test multiple optimization steps to check momentum and bias correction.
        """
        # Initialize parameters
        w_custom = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        w_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        torch_adam = torch.optim.Adam(
            [w_torch], lr=self.lr, betas=(self.b1, self.b2), eps=self.eps
        )

        # Update
        for _ in range(5):
            # Generate random gradients
            grad_custom = cp.random.randn(3).astype(cp.float32)
            grad_torch = torch.from_numpy(cp.asnumpy(grad_custom))

            # Update custom Adam
            w_custom = self.adam.update(grad_custom, w_custom)

            # Update PyTorch Adam
            torch_adam.zero_grad()
            w_torch.grad = grad_torch
            torch_adam.step()

        # Compare results after multiple steps
        np.testing.assert_allclose(
            cp.asnumpy(w_custom),
            w_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Multiple update steps produce different results",
        )

    def test_matrix_update_steps(self):
        """
        Test optimization steps with matrix weights.
        """
        # Initialize matrix parameters
        shape = (3, 4)
        w_custom = cp.ones(shape, dtype=cp.float32)
        w_torch = torch.ones(shape, requires_grad=True)

        # Initialize PyTorch Adam optimizer
        torch_adam = torch.optim.Adam(
            [w_torch], lr=self.lr, betas=(self.b1, self.b2), eps=self.eps
        )

        # Perform multiple update steps to test momentum and bias correction
        for _ in range(5):
            # Generate random gradients
            grad_custom = cp.random.randn(*shape).astype(cp.float32)
            grad_torch = torch.from_numpy(cp.asnumpy(grad_custom))

            # Update custom Adam
            w_custom = self.adam.update(grad_custom, w_custom)

            # Update PyTorch Adam
            torch_adam.zero_grad()
            w_torch.grad = grad_torch
            torch_adam.step()

        # Compare
        np.testing.assert_allclose(
            cp.asnumpy(w_custom),
            w_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Matrix updates unequal",
        )

    def test_zero_gradient(self):
        """
        Test behavior with zero gradients to ensure proper handling of edge cases.
        """
        w_custom = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        grad_custom = cp.zeros_like(w_custom)

        # Perform update with zero gradient
        w_custom_updated = self.adam.update(grad_custom, w_custom)

        # Check that weights changed minimally due to zero gradient
        np.testing.assert_allclose(
            cp.asnumpy(w_custom),
            cp.asnumpy(w_custom_updated),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Zero gradient significantly changed weights",
        )

    def test_nan_gradient_handling(self):
        """
        Test optimizer handles NaN gradients by raising an exception.
        """
        w_custom = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        grad_custom = cp.array([np.nan, 0.2, 0.3], dtype=cp.float32)

        with self.assertRaises(ValueError):
            self.adam.update(grad_custom, w_custom)


if __name__ == "__main__":
    unittest.main()
