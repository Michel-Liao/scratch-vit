import sys
import os
import cupy as cp
from typing import Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from linear import Linear
from class_token import Parameter
from patchify import patchify
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock
from optimizers import Optimizer, Adam


class ViT:
    """
    Vision Transformer
    """

    def __init__(
        self,
        im_dim: Tuple[int, int, int],
        n_patches: int,
        h_dim: int,
        n_heads: int,
        num_blocks: int,
        num_classes: int,
        init_method: str,
    ):
        """Initialize.

        Args:
            im_dim (Tuple[int, int, int]): Image dimensions in the format [C, H, W].
            n_patches (int): Number of patches per row.
            hidden_d (int): Hidden dimension used in the transformer and linear layers.
            n_heads (int): Number of attention heads.
            num_blocks (int): Number of transformer blocks.
            num_classes (int): Number of classes in the dataset.
        """
        self.im_dim = im_dim
        self.n_patches = n_patches
        self.patch_size = (im_dim[1] / n_patches, im_dim[2] / n_patches)
        assert self.patch_size[0] == self.patch_size[1], "Patch size must be square"
        self.input_d = int(im_dim[0] * self.patch_size[0] ** 2)
        self.h_dim = h_dim

        self.linear_proj = Linear(self.input_d, self.h_dim, init_method, bias=False)
        self.class_token = Parameter(cp.random.rand(1, self.h_dim))

        # TODO: FIGURE THIS OUT
        self.pos_embed = PositionalEmbedding(n_patches**2 + 1, h_dim)

        self.blocks = [TransformerBlock(h_dim, n_heads) for _ in range(num_blocks)]

        # MLP head for classification
        self.mlp = Linear(self.h_dim, num_classes)

    def forward(self, images: cp.ndarray) -> cp.ndarray:
        """Forward propagation.

        Args:
            images: input array.

        Returns:
            computed linear layer output.
        """
        patches = patchify(images, int(self.patch_size[0]))
        tokens = self.linear_proj(patches)
        out = cp.stack(
            [cp.vstack((self.class_token.cls, tokens[i])) for i in range(len(tokens))]
        )
        out = self.pos_embed(out)

        for block in self.blocks:
            out = block.forward(out)
        # Class token
        out = self.mlp(out[:, 0])

        return out

    def init_optimizer(self, optimizer: Optimizer) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        self.linear_proj.init_optimizer(optimizer)
        for block in self.blocks:
            block.init_optimizer(optimizer)
        self.mlp.init_optimizer(optimizer)
        self.class_token.init_optimizer(optimizer)

    def backward(self, error: cp.ndarray) -> cp.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        error = self.mlp.backward(error)

        for block in self.blocks[::-1]:
            error = block.backward(error)

        self.class_token.backward(error[:, 0, :])

        removed_cls = error[:, 1:, :]

        _ = self.linear_proj.backward(removed_cls)

        self.class_token.backward(error[:, 0, :])

    def update_params(self) -> None:
        """Update weights based on the calculated gradients."""
        self.mlp.update_params()
        for block in self.blocks[::-1]:
            block.update_params()
        self.linear_proj.update_params()
        self.class_token.update_params()
