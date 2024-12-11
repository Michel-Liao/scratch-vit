import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import cupy as cp
import torch
from typing import Tuple

from model.softmax import Softmax


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.softmax = Softmax()
        self.rtol = 1e-5
        self.atol = 1e-8

        self.test_shapes = [
            (32, 10),  # batch_size=32, features=10
            (16, 100),  # batch_size=16, features=100
        ]

    def test_forward_pass(self):
        """Test forward pass against PyTorch implementation."""
        for shape in self.test_shapes:
            with self.subTest(shape=shape):
                # Generate random input
                x_cp = cp.random.randn(*shape).astype(cp.float32)
                x_torch = torch.from_numpy(cp.asnumpy(x_cp))

                # Compute softmax
                out_cp = self.softmax(x_cp)
                out_torch = torch.nn.functional.softmax(x_torch, dim=-1)

                np.testing.assert_allclose(
                    cp.asnumpy(out_cp),
                    out_torch.numpy(),
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg=f"Forward pass failed for shape {shape}",
                )

                self.assertTrue(
                    cp.allclose(cp.sum(out_cp, axis=-1), 1.0),
                    "Softmax outputs should sum to 1",
                )
                self.assertTrue(
                    cp.all((out_cp >= 0) & (out_cp <= 1)),
                    "Softmax outputs should be between 0 and 1",
                )

    def test_backward_pass(self):
        """Test backward pass against PyTorch implementation."""
        for shape in self.test_shapes:
            with self.subTest(shape=shape):
                # Generate random input and gradient
                x_cp = cp.random.randn(*shape).astype(cp.float32)
                grad_cp = cp.random.randn(*shape).astype(cp.float32)

                # Forward pass
                out_cp = self.softmax(x_cp)

                # Backward pass
                grad_input_cp = self.softmax.backward(grad_cp)

                # PyTorch implementation
                x_torch = torch.from_numpy(cp.asnumpy(x_cp))
                x_torch.requires_grad = True
                grad_torch = torch.from_numpy(cp.asnumpy(grad_cp))

                # Forward pass
                out_torch = torch.nn.functional.softmax(x_torch, dim=-1)
                # Backward pass
                out_torch.backward(grad_torch)
                grad_input_torch = x_torch.grad

                np.testing.assert_allclose(
                    cp.asnumpy(grad_input_cp),
                    grad_input_torch.numpy(),
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg=f"Backward pass failed for shape {shape}",
                )

    def test_numerical_stability(self):
        """Test numerical stability with large inputs."""
        for shape in self.test_shapes:
            with self.subTest(shape=shape):
                x_large = np.random.randn(*shape).astype(np.float32) * 1000
                x_cp = cp.array(x_large)
                x_torch = torch.from_numpy(x_large)

                # Compute softmax
                out_cp = self.softmax(x_cp)
                out_torch = torch.nn.functional.softmax(x_torch, dim=-1)

                np.testing.assert_allclose(
                    cp.asnumpy(out_cp),
                    out_torch.numpy(),
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="Numerical stability test failed",
                )


if __name__ == "__main__":
    unittest.main()
