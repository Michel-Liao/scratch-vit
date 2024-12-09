import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import cupy as cp

from model.patchify import patchify


def ref_patchify(images: cp.ndarray, patch_size: int = 16) -> cp.ndarray:
    """
    Pachify using torch.nn.Unfold.

    Args:
        images (cp.ndarray): Input images of shape (B, C, H, W)
        patch_size (int): Size of square patches

    Returns:
        cp.ndarray: Patches of shape (B, N, patch_size**2 * C)
                   where N = H*W/patch_size**2
    """
    B, C, H, W = images.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), f"Image dimensions {H}x{W} must be divisible by patch_size {patch_size}"

    images_torch = torch.from_numpy(cp.asnumpy(images))

    unfold = nn.Unfold(
        kernel_size=(patch_size, patch_size),
        stride=patch_size,
    )

    patches = unfold(images_torch)  # (B, C*P*P, N)
    patches = patches.transpose(1, 2)  # (B, N, C*P*P)

    return cp.asarray(patches.numpy())


class TestPatch(unittest.TestCase):
    def test_patching(self) -> None:
        images = cp.random.rand(8, 3, 28, 28)  # B, C, H, W
        own_out = patchify(images, 4)
        ref_out = ref_patchify(images, 4)
        np.testing.assert_array_almost_equal(
            own_out.get(), ref_out.get(), 3, "Implementation and reference not equal."
        )


if __name__ == "__main__":
    unittest.main()
