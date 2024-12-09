# scratch_vit

## Install

Run `conda env create -f environment.yml` to install the correct environment.

Run `python -m ipykernel install --user --name=scratch_vit` to install the environment in your Jupyter notebook.

To grab the MNIST data, follow the instructions [here](https://pypi.org/project/python-mnist/).

Note: You need access to a CUDA-enabled GPU.

## TODO
[X] Patchification

[] Position Embedding

[] MLP

[] MHA

[] Layer Norm

[] GELU

[] Optimizer

[] Loss

[] Softmax


## Resources
* [ViT NumPy Implementation](https://github.com/kmsgnnew/vision_transformer_numpy/tree/main)
* [CuPy (NumPy for GPU)](https://cupy.dev/)
* [Numba (fast Python compilter)](https://numba.pydata.org/)

## Observations:
* When doing patchify, issue of what happens if the image dimension doesn't work well with the patch dimension? The ViT paper doesn't explain this case but the Appendix B.1 shows they use resolution 224 x 224 which is divisible by their patch sizes of 16 and 32. We will do the same.