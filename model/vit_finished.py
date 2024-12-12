import cupy as cp
from typing import Tuple

from linear import Linear
from class_token import Parameter
from patchify import patchify
from positional_embedding import PositionalEmbedding
from vit_modules import TransformerBlock


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
        classes: int,
    ):
        """Initialize.

        Args:
            im_dim (Tuple[int, int, int]): Image dimensions in the format [C, H, W].
            n_patches (int): Number of patches in the image.
            hidden_d (int): Hidden dimension used in the transformer and linear layers.
            n_heads (int): Number of attention heads.
            num_blocks (int): Number of transformer blocks.
            classes (int): Number of classes in the dataset.
        """
        self.im_dim = im_dim
        self.n_patches = n_patches
        self.patch_size = (im_dim[1] / n_patches, im_dim[2] / n_patches)
        assert self.patch_size[0] == self.patch_size[1], "Patch size must be square"
        self.input_d = int(im_dim[0] * self.patch_size[0] ** 2)
        self.h_dim = h_dim

        self.linear_proj = Linear(self.input_d, self.h_dim, bias=False)
        self.class_token = Parameter(cp.random.rand(1, self.h_dim))

        self.pos_embed = patchify(self.n_patches**2 + 1, self.h_dim)

        self.blocks = [TransformerBlock(h_dim, n_heads) for _ in range(num_blocks)]

        # MLP head for classification
        self.mlp = Linear(self.h_dim, classes)

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            images: input array.

        Returns:
            computed linear layer output.
        """
        patches = convert_image_to_patches(images, self.n_patches)
        tokens = self.linear_proj(patches)
        out = np.stack(
            [np.vstack((self.class_token.val, tokens[i])) for i in range(len(tokens))]
        )
        out = out + self.pos_embed
        for block in self.blocks:
            out = block.forward(out)
        out = self.mlp(out[:, 0])
        return out

    def set_optimizer(self, optimizer_algo: object) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        self.linear_proj.set_optimizer(optimizer_algo)
        for block in self.blocks:
            block.set_optimizer(optimizer_algo)
        self.mlp.set_optimizer(optimizer_algo)
        self.class_token.set_optimizer(optimizer_algo)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        error = self.mlp.backward(error)

        for block in self.blocks[::-1]:
            error = block.backward(error)
        removed_cls = error[:, 1:, :]
        _ = self.linear_proj.backward(removed_cls)
        self.class_token.backward(error[:, 0, :])

    def update_weights(self) -> None:
        """Update weights based on the calculated gradients."""
        self.mlp.update_weights()
        for block in self.blocks[::-1]:
            block.update_weights()
        self.linear_proj.update_weights()
        self.class_token.update_weights()
