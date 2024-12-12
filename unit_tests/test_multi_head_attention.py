import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import torch
import cupy as cp
from torch import nn
import torch.nn.functional as F

from model.multi_head_attention import (
    MultiHeadAttention,
)


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.batch_size = 2
        self.seq_length = 4
        self.dimension = 6
        self.n_heads = 2
        self.mha = MultiHeadAttention(dimension=self.dimension, n_heads=self.n_heads)

        # Create sample input
        np.random.seed(42)
        # self.input_data = cp.array(
        #     np.random.randn(self.batch_size, self.seq_length, self.dimension)
        # )
        self.input_data = cp.array(np.random.randn(self.batch_size, self.dimension))

    def test_initialization(self):
        """Test if the MultiHeadAttention layer is initialized correctly"""
        self.assertEqual(self.mha.n_heads, self.n_heads)
        self.assertEqual(self.mha.d_head, self.dimension // self.n_heads)
        self.assertEqual(len(self.mha.q_mappings), self.n_heads)
        self.assertEqual(len(self.mha.k_mappings), self.n_heads)
        self.assertEqual(len(self.mha.v_mappings), self.n_heads)
        self.assertEqual(len(self.mha.softmax), self.n_heads)

    def test_forward_shape(self):
        """Test if forward pass produces correct output shape"""
        output = self.mha.forward(self.input_data)
        expected_shape = (self.batch_size, self.seq_length, self.dimension)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_deterministic(self):
        """Test if forward pass is deterministic"""
        output1 = self.mha.forward(self.input_data)
        output2 = self.mha.forward(self.input_data)
        cp.testing.assert_array_almost_equal(output1, output2)

    def test_backward_shape(self):
        """Test if backward pass produces correct gradient shape"""
        # Forward pass first
        output = self.mha.forward(self.input_data)

        # Create dummy gradient
        grad = cp.ones_like(output)

        # Backward pass
        breakpoint()
        input_grad = self.mha.backward(grad)

        # Check if gradient shape matches input shape
        self.assertEqual(input_grad.shape, self.input_data.shape)

    def test_attention_mask(self):
        """Test if attention scores are properly normalized"""
        output = self.mha.forward(self.input_data)

        # Check if attention scores sum to 1 along the appropriate dimension
        attention_scores = self.mha.attention_seqs

        # Split attention scores by head
        attention_heads = cp.split(attention_scores, self.n_heads, axis=-1)

        for head_scores in attention_heads:
            # Sum along the sequence length dimension (dim 2)
            sums = cp.sum(head_scores, axis=2)
            # Check if sums are close to 1
            cp.testing.assert_array_almost_equal(sums, cp.ones_like(sums), decimal=5)

    # def test_zero_input(self):
    #     """Test behavior with zero input"""
    #     zero_input = cp.zeros_like(self.input_data)
    #     output = self.mha.forward(zero_input)

    #     # Output should not be all zeros due to learned weights
    #     self.assertFalse(cp.allclose(output, cp.zeros_like(output)))

    # def test_backward_zero_grad(self):
    #     """Test backward pass with zero gradient"""
    #     # Forward pass
    #     output = self.mha.forward(self.input_data)

    #     # Backward pass with zero gradient
    #     zero_grad = cp.zeros_like(output)
    #     input_grad = self.mha.backward(zero_grad)

    #     # Input gradient should be zero
    #     cp.testing.assert_array_almost_equal(input_grad, cp.zeros_like(input_grad))


if __name__ == "__main__":
    unittest.main()
