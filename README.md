# scratch_vit

## Install

Run `conda env create -f environment.yml` to install the correct environment.

Run `python -m ipykernel install --user --name=scratch_vit` to install the environment in your Jupyter notebook.

To grab the MNIST data, follow the instructions [here](https://pypi.org/project/python-mnist/).

Note: You need access to a CUDA-enabled GPU.

## TODO
- [X] Patchification
- [ ] Position Embedding
- [ ] MLP
- [ ] MHA
- [ ] Layer Norm
- [ ] GELU
- [X] Optimizer
- [ ] Loss
- [ ] Softmax


## Resources
* [ViT NumPy Implementation](https://github.com/kmsgnnew/vision_transformer_numpy/tree/main)
* [Transformer NumPy Implementation](https://github.com/AkiRusProd/numpy-transformer/tree/master)
* [CuPy (NumPy for GPU)](https://cupy.dev/)
* [Numba (fast Python compilter)](https://numba.pydata.org/)

## Citations
* ViT Paper
* Transformer paper
* Adam paper
* Kaiming He Init Paper
* Xavier Init Paper

## Observations
* When doing patchify, issue of what happens if the image dimension doesn't work well with the patch dimension? The ViT paper doesn't explain this case but the Appendix B.1 shows they use resolution 224 x 224 which is divisible by their patch sizes of 16 and 32. We will do the same.
* Implementing things in the order of the paper's method section isn't possible. Often, earlier steps mentioned, like learned positional embedding, require something later on, in this case an MLP.
* Adam optimizer actually can optimizer with respect to a weight matrix w that has the bias concatenated so we can write less code.
* Lots of minor design decisions, e.g. type of optimizers, ways to allocate memory in functions, dimensions to store weights matrix in linear layer ([out, in] better than [in, out] for the backward pass because of some CUDA and caching reasons).
* Learned about ABC and unittest libraries in this.