import cupy as cp


def patchify(images: cp.ndarray, patch_size: int = 16) -> cp.ndarray:
    """
    Patchify the image. Requires that H and W is divisible by patch_size.

    Args:
        images (cp.ndarray): (B, C, H, W)
        patch_size (int): Size of the square patches.

    Returns:
        patches (cp.ndarray): (B, N, patch_size**2 * C), where N = H*W/patch_size**2
    """
    B, C, H, W = images.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), f"Image dimensions {H}x{W} must be divisible by patch_size {patch_size}"

    num_patches_in_h = H // patch_size
    num_patches_in_w = W // patch_size

    patches = cp.empty(
        (B, H * W // patch_size**2, C * patch_size**2), dtype=images.dtype
    )

    for y in range(num_patches_in_h):
        for x in range(num_patches_in_w):
            patch = images[
                :,
                :,
                y * patch_size : (y + 1) * patch_size,
                x * patch_size : (x + 1) * patch_size,
            ]
            patches[:, num_patches_in_w * y + x, :] = patch.reshape(
                B, C * patch_size * patch_size
            )
    return patches
