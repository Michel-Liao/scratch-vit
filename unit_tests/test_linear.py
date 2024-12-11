import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import cupy as cp

from model.linear import Linear


class TestLinear(unittest.TestCase):
    def setUp(self) -> None:
        """
        Init with same random seeds and optimizer parameters for reproducibility.
        """
        np.random.seed(16)
        torch.manual_seed(16)
        cp.random.seed(16)

        self.batch_size = 32
        self.in_features = 64
        self.out_features = 128

    def test_forward_pass(self):
        """Test if forward pass produces same output as PyTorch's nn.Linear"""
        # Create random input
        x_np = np.random.randn(self.batch_size, self.in_features)
        x_torch = torch.FloatTensor(x_np)
        x_cupy = cp.array(x_np)

        # Initialize implementations
        custom_linear = Linear(self.in_features, self.out_features)
        torch_linear = nn.Linear(self.in_features, self.out_features)

        # Copy weights and biases to PyTorch implementation
        with torch.no_grad():
            torch_linear.weight.data = torch.FloatTensor(
                cp.asnumpy(custom_linear.weight)
            )
            torch_linear.bias.data = torch.FloatTensor(cp.asnumpy(custom_linear.bias))

        # Compute outputs
        custom_output = cp.asnumpy(custom_linear(x_cupy))
        torch_output = torch_linear(x_torch).detach().numpy()

        # Compare outputs
        np.testing.assert_allclose(custom_output, torch_output, rtol=1e-5, atol=1e-5)

    def test_backward_pass(self):
        """Test if backward pass computes correct gradients compared to PyTorch"""

        # Create random input and gradient
        x_cp = cp.random.randn(self.batch_size, self.in_features)
        grad_z_cp = cp.random.randn(self.batch_size, self.out_features)

        x_torch = torch.FloatTensor(cp.asnumpy(x_cp))
        x_torch.requires_grad = True
        grad_z_torch = torch.FloatTensor(cp.asnumpy(grad_z_cp))

        # Initialize implementations
        custom_linear = Linear(self.in_features, self.out_features)
        torch_linear = nn.Linear(self.in_features, self.out_features)

        # Copy weights and biases to PyTorch implementation
        with torch.no_grad():
            torch_linear.weight.data = torch.FloatTensor(
                cp.asnumpy(custom_linear.get_weight())
            )
            torch_linear.bias.data = torch.FloatTensor(
                cp.asnumpy(custom_linear.get_bias())
            )

        torch_output = torch_linear(x_torch)

        # Forward pass to cache input
        _ = custom_linear(x_cp)

        # Backward pass
        custom_grad_x = custom_linear.backward(grad_z_cp)
        torch_output.backward(gradient=grad_z_torch)

        # Get gradients
        custom_grad_w, custom_grad_b = custom_linear.get_grads()
        torch_grad_w = torch_linear.weight.grad.numpy()
        torch_grad_b = torch_linear.bias.grad.numpy()

        # Compare gradients
        np.testing.assert_allclose(
            cp.asnumpy(custom_grad_x), x_torch.grad.numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            cp.asnumpy(custom_grad_w), torch_grad_w, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            cp.asnumpy(custom_grad_b), torch_grad_b, rtol=1e-5, atol=1e-5
        )

    def test_no_bias(self):
        """Test linear layer without bias"""
        custom_linear = Linear(self.in_features, self.out_features, bias=False)
        torch_linear = nn.Linear(self.in_features, self.out_features, bias=False)

        # Check that bias is None
        self.assertIsNone(custom_linear.bias)

        # Test forward pass without bias
        x_np = np.random.randn(self.batch_size, self.in_features)
        x_torch = torch.FloatTensor(x_np)
        x_cupy = cp.array(x_np)

        with torch.no_grad():
            torch_linear.weight.data = torch.FloatTensor(
                cp.asnumpy(custom_linear.weight)
            )

        custom_output = cp.asnumpy(custom_linear(x_cupy))
        torch_output = torch_linear(x_torch).detach().numpy()

        np.testing.assert_allclose(custom_output, torch_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
