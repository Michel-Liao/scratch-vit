import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import cupy as cp
from src.optimizers import Adam
from src.positional_embedding import PositionalEmbedding


class TestPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        """
        Set up test parameters
        """
        self.num_patches = 16
        self.embedding_dim = 32
        self.batch_size = 8

    def test_initialization(self):
        """
        Test different initialization methods
        """
        # Normal initialization
        pos_embed_normal = PositionalEmbedding(
            num_patches=self.num_patches,
            embedding_dim=self.embedding_dim,
            init="normal",
        )
        self.assertEqual(
            pos_embed_normal.pos_embedding.shape,
            (1, self.num_patches, self.embedding_dim),
        )

        # Uniform initialization
        pos_embed_uniform = PositionalEmbedding(
            num_patches=self.num_patches,
            embedding_dim=self.embedding_dim,
            init="uniform",
        )
        self.assertEqual(
            pos_embed_uniform.pos_embedding.shape,
            (1, self.num_patches, self.embedding_dim),
        )

        # Invalid initialization
        with self.assertRaises(ValueError):
            PositionalEmbedding(
                num_patches=self.num_patches,
                embedding_dim=self.embedding_dim,
                init="invalid",
            )

    def test_forward_pass(self):
        """
        Test forward pass with valid and invalid inputs
        """
        pos_embed = PositionalEmbedding(
            num_patches=self.num_patches, embedding_dim=self.embedding_dim
        )

        # Test valid input
        x = cp.random.normal(
            0, 1, (self.batch_size, self.num_patches, self.embedding_dim)
        )
        output = pos_embed.forward(x)
        self.assertEqual(output.shape, x.shape)

        # Input type validation
        with self.assertRaises(TypeError):
            pos_embed.forward(
                np.random.normal(
                    0, 1, (self.batch_size, self.num_patches, self.embedding_dim)
                )
            )

        # Input dimensionality validation
        with self.assertRaises(ValueError):
            pos_embed.forward(
                cp.random.normal(0, 1, (self.batch_size, self.num_patches))
            )

        # Input shape validation
        with self.assertRaises(ValueError):
            pos_embed.forward(
                cp.random.normal(
                    0, 1, (self.batch_size, self.num_patches + 1, self.embedding_dim)
                )
            )

    def test_backward_pass(self):
        """Test backward pass and gradient computation"""
        pos_embed = PositionalEmbedding(
            num_patches=self.num_patches, embedding_dim=self.embedding_dim
        )

        # Forward pass to set up cache
        x = cp.random.normal(
            0, 1, (self.batch_size, self.num_patches, self.embedding_dim)
        )
        pos_embed.forward(x)

        # Test backward pass
        grad_output = cp.random.normal(
            0, 1, (self.batch_size, self.num_patches, self.embedding_dim)
        )
        grad_input = pos_embed.backward(grad_output)

        # Check gradient shapes
        self.assertEqual(grad_input.shape, x.shape)
        self.assertEqual(pos_embed.get_grads().shape, pos_embed.pos_embedding.shape)

        # Check if gradients are summed across batch dimension
        expected_grad_shape = (1, self.num_patches, self.embedding_dim)
        self.assertEqual(pos_embed.grad_pos_embedding.shape, expected_grad_shape)

    def test_parameter_update(self):
        """Test parameter updates with optimizer"""
        pos_embed = PositionalEmbedding(
            num_patches=self.num_patches, embedding_dim=self.embedding_dim
        )

        # Store initial parameters
        initial_params = cp.copy(pos_embed.get_pos_embedding())

        # Forward and backward pass
        x = cp.random.normal(
            0, 1, (self.batch_size, self.num_patches, self.embedding_dim)
        )
        pos_embed.forward(x)
        grad_output = cp.random.normal(0, 1, x.shape)
        pos_embed.backward(grad_output)

        pos_embed.update_params()

        # Check if parameters have been updated
        updated_params = pos_embed.get_pos_embedding()
        self.assertFalse(cp.allclose(initial_params, updated_params))


if __name__ == "__main__":
    unittest.main()
