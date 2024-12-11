import sys
import os
import unittest
from typing import Type, Tuple, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.autograd import gradcheck
import numpy as np
import cupy as cp

from model.layer_norm import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        """
        Set random seeds and parameters.
        """
        torch.manual_seed(16)
        np.random.seed(16)
        cp.random.seed(16)

        # Parameters
        self.batch_size = 32
        self.seq_length = 10
        self.hidden_size = 64
        self.eps = 1e-5

        self.custom_ln = LayerNorm(self.hidden_size, eps=self.eps)
        self.torch_ln = nn.LayerNorm(self.hidden_size, eps=self.eps)

    def test_forward_pass(self):
        # Generate random input
        x_cp = cp.random.randn(self.batch_size, self.seq_length, self.hidden_size)
        x_torch = torch.FloatTensor(cp.asnumpy(x_cp))

        # Forward passes
        custom_output = self.custom_ln(x_cp)
        torch_output = self.torch_ln(x_torch)

        np.testing.assert_allclose(
            cp.asnumpy(custom_output),
            torch_output.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Forward pass outputs do not match",
        )

    def test_backward_pass(self):
        x_cp = cp.random.randn(self.batch_size, self.seq_length, self.hidden_size)
        grad_cp = cp.random.randn(self.batch_size, self.seq_length, self.hidden_size)

        # Convert to respective formats
        x_torch = torch.FloatTensor(cp.asnumpy(x_cp)).requires_grad_(True)
        grad_torch = torch.FloatTensor(cp.asnumpy(grad_cp))

        # Forward and backward passes
        torch_output = self.torch_ln(x_torch)
        torch_output.backward(grad_torch)
        torch_grad = x_torch.grad.numpy()

        custom_output = self.custom_ln(x_cp)
        custom_grad = self.custom_ln.backward(grad_cp)
        custom_grad_np = cp.asnumpy(custom_grad)

        np.testing.assert_allclose(
            custom_grad_np,
            torch_grad,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Backward pass gradients do not match",
        )

    # def test_gradcheck(self):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     batch_size = 4
    #     seq_length = 3
    #     hidden_size = 8

    #     # Create torch layer norm for gradcheck
    #     layer_norm = nn.LayerNorm(hidden_size, eps=self.eps).to(device)

    #     # Generate random input
    #     x = torch.randn(
    #         batch_size,
    #         seq_length,
    #         hidden_size,
    #         requires_grad=True,
    #     ).to(device)

    #     # Perform gradcheck
    #     self.assertTrue(
    #         gradcheck(layer_norm, x, eps=1e-6, atol=1e-4),
    #         "gradcheck failed for LayerNorm",
    #     )

    def test_edge_cases(self):
        shapes = [
            (1, self.hidden_size),  # Single sample
            (100, 1, self.hidden_size),  # Different batch size
            (self.batch_size, 1, self.hidden_size),  # Sequence length 1
        ]

        for shape in shapes:
            x_cp = cp.random.randn(*shape)
            x_torch = torch.FloatTensor(cp.asnumpy(x_cp))

            custom_output = self.custom_ln(x_cp)
            torch_output = self.torch_ln(x_torch)

            np.testing.assert_allclose(
                cp.asnumpy(custom_output),
                torch_output.detach().numpy(),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Forward pass failed for shape {shape}",
            )

    def test_parameter_gradients(self):
        # Test gradients for gain and bias parameters
        x_cp = cp.random.randn(self.batch_size, self.seq_length, self.hidden_size)
        grad_cp = cp.random.randn(self.batch_size, self.seq_length, self.hidden_size)

        # PyTorch
        x_torch = torch.FloatTensor(cp.asnumpy(x_cp)).requires_grad_(True)
        grad_torch = torch.FloatTensor(cp.asnumpy(grad_cp))

        torch_output = self.torch_ln(x_torch)
        torch_output.backward(grad_torch)

        # Custom implementation
        custom_output = self.custom_ln(x_cp)
        self.custom_ln.backward(grad_cp)

        # Get parameter gradients
        custom_grad_g, custom_grad_b = self.custom_ln.get_grads()
        torch_grad_weight = self.torch_ln.weight.grad.numpy()
        torch_grad_bias = self.torch_ln.bias.grad.numpy()

        # Compare parameter gradients
        np.testing.assert_allclose(
            cp.asnumpy(custom_grad_g),
            torch_grad_weight,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Weight gradients do not match",
        )

        np.testing.assert_allclose(
            cp.asnumpy(custom_grad_b),
            torch_grad_bias,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Bias gradients do not match",
        )


if __name__ == "__main__":
    unittest.main()
