import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cupy as cp
import copy

from src.optimizers import Optimizer, Adam


class PositionalEmbedding:
    """
    Learnable positional embeddings.
    """

    def __init__(
        self,
        num_patches: int,
        embedding_dim: int,
        init: str = "normal",
    ) -> None:
        """
        Initialize positional embeddings.

        Args:
            num_patches (int): Number of patches (sequence length)
            embedding_dim (int): Dimension of embeddings
            init (str): Initialization method for embeddings
        """
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

        # Initialize learnable position embeddings
        if init == "normal":
            self.pos_embedding = cp.random.normal(
                0, 0.02, (1, num_patches, embedding_dim)
            )
        elif init == "uniform":
            self.pos_embedding = cp.random.uniform(
                -0.02, 0.02, (1, num_patches, embedding_dim)
            )
        else:
            raise ValueError(f"Unknown initialization method: {init}")

        self.cache = {}

        self._init_optimizer(optimizer=Adam())

    def _init_optimizer(self, optimizer: Optimizer) -> None:
        """
        Initialize optimizer for positional embeddings.

        Args:
            optimizer (Optimizer): Optimizer object
        """
        self.optimizer = copy.deepcopy(optimizer)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Add positional embeddings to the input sequence.

        Args:
            x (cp.ndarray): Input tensor of shape [batch_size, num_patches, embedding_dim]

        Returns:
            cp.ndarray: Output tensor with positional embeddings added
        """
        if not isinstance(x, cp.ndarray):
            raise TypeError(f"Expected cupy.ndarray, got {type(x)}")

        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")

        if x.shape[1:] != (self.num_patches, self.embedding_dim):
            raise ValueError(
                f"Expected input shape [batch_size, {self.num_patches}, {self.embedding_dim}], "
                f"got {x.shape}"
            )

        self.cache["input"] = x

        # Broadcasting handles batch dimension
        out = x + self.pos_embedding

        return out

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """
        Backward pass for positional embeddings.

        Args:
            grad_output (cp.ndarray): Gradient of loss with respect to output

        Returns:
            cp.ndarray: Gradient of loss with respect to input
        """
        # Sum across batch dimension since pos_embedding is shared
        self.grad_pos_embedding = cp.sum(grad_output, axis=0, keepdims=True)

        return grad_output

    def update_params(self) -> None:
        """
        Update positional embedding parameters using optimizer.
        """
        self.pos_embedding = self.optimizer.update(
            self.grad_pos_embedding, self.pos_embedding
        )

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def get_pos_embedding(self) -> cp.ndarray:
        """
        Get current positional embeddings.

        Returns:
            cp.ndarray: Positional embeddings
        """
        return self.pos_embedding

    def get_grads(self) -> cp.ndarray:
        """
        Get gradients of positional embeddings.

        Returns:
            cp.ndarray: Gradients of positional embeddings
        """
        return self.grad_pos_embedding

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)
