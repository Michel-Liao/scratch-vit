import cupy as cp


def patchify(images: cp.ndarray, patch_size: int = 16) -> cp.ndarray:
    """
    Convert an image into patches for tokenization. This requires that H and W
    is evenly divisible by the patch_size.

    Args:
        images (cp.ndarray): Vector of form (batches, channels, height, width).
        patch_size (int): Size of the square patches.

    Returns:
        patches (cp.ndarray): (B, N, patch_size**2 * C), where N = H*W/patch_size**2
    """

    assert type(images) == cp.ndarray, "Input must be a cupy array"

    B, C, H, W = images.shape

    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), f"Image dimensions {H}x{W} must be divisible by patch_size {patch_size}"

    num_h_patches = H // patch_size
    num_w_patches = W // patch_size

    patches = cp.empty(
        (B, H * W // patch_size**2, C * patch_size**2), dtype=images.dtype
    )

    for y in range(num_h_patches):
        for x in range(num_w_patches):
            patch = images[
                :,
                :,
                y * patch_size : (y + 1) * patch_size,
                x * patch_size : (x + 1) * patch_size,
            ]
            patches[:, num_w_patches * y + x, :] = patch.reshape(
                B, C * patch_size * patch_size
            )

    return patches
