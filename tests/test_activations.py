import sys
import os
import unittest
from typing import Type, Tuple, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import cupy as cp

from src.activations import ReLU, FastGELU, GELU, Activation


class TestActivationFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set random seeds, test inputs, and gradients.
        """
        np.random.seed(16)
        torch.manual_seed(16)
        cp.random.seed(16)

        self.batch_size = 32
        self.features = 64

        self.test_inputs = cp.array(np.random.randn(self.batch_size, self.features))
        self.test_grads = cp.array(np.random.randn(self.batch_size, self.features))

    def _compare_with_pytorch(
        self,
        custom_activation: Type[Activation],
        pytorch_activation: Callable,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Tuple[bool, bool]:
        """
        Compares custom activation implementation with PyTorch implementation.

        Args:
            custom_activation (Type[Activation]): Custom activation class.
            pytorch_activation (Callable): PyTorch activation function.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.

        Returns:
            Tuple[bool, bool]: Match for forward and backward pass.
        """
        activation = custom_activation()

        # Convert CuPy array to PyTorch tensor
        torch_input = torch.tensor(cp.asnumpy(self.test_inputs), requires_grad=True)

        # Forward pass
        custom_output = activation.forward(self.test_inputs)
        torch_output = pytorch_activation(torch_input)

        forward_match = cp.allclose(
            custom_output, cp.array(torch_output.detach().numpy()), rtol=rtol, atol=atol
        )

        # Backward pass
        custom_grad = activation.backward(self.test_grads)

        torch_grad = torch.tensor(cp.asnumpy(self.test_grads))
        torch_output.backward(torch_grad)
        torch_input_grad = torch_input.grad

        backward_match = cp.allclose(
            custom_grad, cp.array(torch_input_grad.numpy()), rtol=rtol, atol=atol
        )

        return forward_match, backward_match

    def test_relu_forward(self):
        """
        Test ReLU forward pass matches PyTorch
        """
        forward_match, _ = self._compare_with_pytorch(ReLU, torch.nn.ReLU())
        self.assertTrue(forward_match, "ReLU forward pass does not match PyTorch")

    def test_relu_backward(self):
        """
        Test ReLU backward pass matches PyTorch
        """
        _, backward_match = self._compare_with_pytorch(ReLU, torch.nn.ReLU())
        self.assertTrue(backward_match, "ReLU backward pass does not match PyTorch")

    def test_relu_zero_input(self):
        """
        Test ReLU with zero input
        """
        activation = ReLU()
        zero_input = cp.zeros((self.batch_size, self.features))
        output = activation.forward(zero_input)
        self.assertTrue(cp.all(output == 0), "ReLU fails for zero input")

    def test_gelu_forward(self):
        """
        Test GELU forward pass matches PyTorch
        """
        forward_match, _ = self._compare_with_pytorch(
            GELU,
            torch.nn.GELU(),
            rtol=1e-4,
            atol=1e-7,
        )
        self.assertTrue(forward_match, "GELU forward pass does not match PyTorch")

    def test_gelu_backward(self):
        """
        Test GELU backward pass matches PyTorch
        """
        _, backward_match = self._compare_with_pytorch(
            GELU, torch.nn.GELU(), rtol=1e-4, atol=1e-7
        )
        self.assertTrue(
            backward_match, "GELU backward pass does not match PyTorch implementation"
        )

    def test_fast_gelu_forward(self):
        """
        Test FastGELU forward pass matches PyTorch GELU
        """
        forward_match, _ = self._compare_with_pytorch(
            FastGELU,
            torch.nn.GELU(),
            rtol=1e-3,
            atol=1e-3,
        )
        self.assertTrue(
            forward_match, "FastGELU forward pass deviates too much from PyTorch"
        )

    def test_fast_gelu_backward(self):
        """
        Test FastGELU backward pass matches PyTorch GELU
        """
        _, backward_match = self._compare_with_pytorch(
            FastGELU,
            torch.nn.GELU(),
            rtol=1e-3,
            atol=1e-3,
        )
        self.assertTrue(
            backward_match, "FastGELU backward pass deviates too much from PyTorch"
        )

    def test_activation_shapes(self):
        """Test that activations preserve input shapes"""
        activations = [ReLU(), GELU(), FastGELU()]
        test_shapes = [(1, 10), (5, 5), (32, 64, 64)]

        for activation in activations:
            for shape in test_shapes:
                test_input = cp.random.randn(*shape)
                output = activation.forward(test_input)
                self.assertEqual(
                    output.shape,
                    test_input.shape,
                    f"{activation.__class__.__name__} changed input shape",
                )

    def test_activation_dtype_preservation(self):
        """Test that activations preserve input dtype"""
        activations = [ReLU(), GELU(), FastGELU()]
        dtypes = [cp.float32, cp.float64]

        for activation in activations:
            for dtype in dtypes:
                test_input = cp.random.randn(10, 10).astype(dtype)
                output = activation.forward(test_input)
                self.assertEqual(
                    output.dtype,
                    dtype,
                    f"{activation.__class__.__name__} changed dtype",
                )


if __name__ == "__main__":
    unittest.main()
