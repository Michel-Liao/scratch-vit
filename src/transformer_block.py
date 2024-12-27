import sys
import os
import cupy as cp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import Optimizer, Adam
from activations import GeLU
from layer_norm import LayerNorm
from linear import Linear
from multi_head_attention import MultiHeadAttention


class TransformerBlock:
    """
    Transformer module as shown in Figure 1 in the ViT paper.
    """

    def __init__(self, h_dim: int, n_heads: int, mlp_ratio: int = 4) -> None:
        """Initialize the ViT module.

        Args:
            h_dim (int): Hidden dimension of the embedded patches.
            n_heads: Number of attention heads.
            mlp_ratio: Multiplicative factor of the hidden dimension in the MLP layer. Defaults to 4 as per the ViT paper.
        """
        self.h_dim = h_dim
        self.n_heads = n_heads

        # (Rough) module architecture
        self.layer_norm_1 = LayerNorm(h_dim)
        self.mha = MultiHeadAttention(h_dim, n_heads)
        self.layer_norm_2 = LayerNorm(h_dim)
        self.mlp_1 = Linear(h_dim, mlp_ratio * h_dim)
        self.gelu = GeLU()
        self.mlp_2 = Linear(mlp_ratio * h_dim, h_dim)
        self.cache = dict(input=None)
        self.optimizer = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Forward propagation. Follows equations 1-4 in the paper.

        Args:
            x (cp.ndarray): Input tensor.

        Returns:
            cp.ndarray: Output tensor.
        """
        # Cache for backprop
        self.cache = dict(input=x)

        mha_out = self.mha.forward(self.layer_norm_1(x))
        stage_1_out = x + mha_out

        out = self.layer_norm_2(stage_1_out)
        out = self.mlp_1(out)
        out = self.gelu(out)
        out = self.mlp_2(out)
        out = out + stage_1_out

        return out

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """Backward propagation. Reverse of forward pass.

        Args:
            grad (cp.ndarray): Gradient with respect to the output.

        Returns:
            cd.ndarray: Gradient with respect to the input.
        """
        # Cache for backprop
        x = self.cache["input"]

        # For last transformer block, grad only contains CLS token gradient
        if grad.shape != x.shape:
            grad_in = cp.zeros(x.shape)
            grad_in[:, 0] = grad
        else:
            grad_in = grad

        grad_in_skip = grad_in

        # Backprop
        grad_in = self.mlp_2.backward(grad_in)
        grad_in = self.gelu.backward(grad_in)
        grad_in = self.mlp_1.backward(grad_in)
        grad_in = self.layer_norm_2.backward(grad_in)
        grad_in = grad_in + grad_in_skip

        grad_in_skip = grad_in
        grad_in = self.mha.backward(grad_in)
        grad_in = self.layer_norm_1.backward(grad_in)
        grad_in = grad_in + grad_in_skip

        return grad_in

    def init_optimizer(self, optimizer: Optimizer) -> None:
        """
        Initialize optimizers.
        """
        self.layer_norm_1.init_optimizer(optimizer)
        self.layer_norm_2.init_optimizer(optimizer)
        self.mha.init_optimizer(optimizer)
        self.mlp_1.init_optimizer(optimizer)
        self.mlp_2.init_optimizer(optimizer)

    def update_params(self) -> None:
        """
        Update the parameters based on gradients.
        """
        self.layer_norm_1.update_params()
        self.layer_norm_2.update_params()
        self.mha.update_params()
        self.mlp_1.update_params()
        self.mlp_2.update_params()

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)
