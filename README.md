# Vision Transformer from Scratch

![CuPy Badge](https://img.shields.io/badge/CuPy-005BAC?logo=numpy&logoColor=fff&style=flat)
![Unittest Badge](https://img.shields.io/badge/Unittest-239120?logo=python&logoColor=fff&style=flat)

This repository contains a from-scratch re-implementation of the Vision Transformer,
first introduced by Dosovitskiy et. al. in 2020. The source code is written using
CuPy, a CUDA-enabled sister library to NumPy. All modules and processes of the 
Vision Transformer are re-implemented faithful to their description in the paper, 
not to any other versions. More information on all modules and processes can be
found in the [`src` folder](./src/).

To verify the implementation of our model, we run unit tests against each
component that we implemented. Thus, every file/component in the `src` folder
has a corresponding unit test file in the `tests` folder that compares the outputs
of our from-scratch implementation against PyTorch's implementation. All components
were verified up to a precision of three decimal places.

## Getting Started

This repository is organized by the following:

```
SCRATCH_VIT/
├── data/           (Where datasets are to be stored)
├── runs/           (Where run outputs and logs are stored)
├── scripts/        (Bash scripts to automate processes such as hyperparameter tuning)
├── src/            (Source code for our from-scratch implementation)
├── tests/          (Unit test files to verify against PyTorch)
├── visualize/      (Files useful to visualize graphs, images, etc.)
├── preprocess.py   (Python script to prepare data)
├── trainer.py      (Python script that houses the ViT trainer class)
└── environment.yml (Library environment file)
└── ...
```

### Environment Setup

**Note:** You need access to a CUDA-enabled GPU to run the code.

- Run `conda env create -f environment.yml` to install the correct environment.
- Run `python -m ipykernel install --user --name=scratch_vit` to install the environment in your Jupyter notebook.

### Download and Preprocess the Data

This repoistory has been initially designed to work with the MNIST and CIFAR10
datasets. To begin, run the following command from the root directory.

```
python ./preprocess.py
```

It's possible to add additional datasets. Reference the `preprocess.py` to 
better understand what are the preprocessing steps required. In short, it's
required that the dataset be cut into three .npy files, one for train, validation,
and test. Each .npy file should contain two numpy arrays, the input in shape
`(num_samples, channels, img_height, img_width)` and the labels in shape 
`(num_samples, label_id)`. The label ids should be one-hot encoded.

### Training the model

The Vision Transformer can be trained using the `trainer.py` script. The script accepts various command-line arguments to customize the training process:

#### Data Parameters
- `--data_path`: (Required) Path to the preprocessed dataset files. The script expects three files with suffixes "_train", "_val", and "_test"
- `--batch_size`: Number of samples per training batch (default: 16)
- `--epochs`: Total number of training epochs (default: 5)
- `--eval_interval`: Number of epochs between validation evaluations (default: 2)

#### Model Architecture
- `--hidden_dim`: Dimensionality of the transformer's hidden layers (default: 128)
- `--num_heads`: Number of attention heads in multi-head attention (default: 4)
- `--num_blocks`: Number of transformer encoder blocks (default: 4)
- `--patch_size`: Size of image patches (default: 7)

#### Optimization Parameters
- `--learning_rate`: Learning rate for the Adam optimizer (default: 1e-9)
- `--init_method`: Weight initialization method, choices: ["he", "normal", "uniform", "xavier"] (default: "he")

#### Example Usage
```
python trainer.py --data_path ./data --batch_size 32 --epochs 10 --eval_interval 1 --hidden_dim 256 --num_heads 8 --num_blocks 6 --patch_size 16 --learning_rate 1e-4 --init_method xavier
```

### Unit Tests

To run the unit tests:

1. `cd unit_tests`
2. `python -m unittest [scriptname].py`

## Citations

- ViT Paper
- Transformer paper
- Adam paper
- Kaiming He Init Paper
- Xavier Init Paper
- GELU Paper
- RELU Paper
- Layer Norm Paper

## Notes

- When doing patchify, issue of what happens if the image dimension doesn't work well with the patch dimension? The ViT paper doesn't explain this case but the Appendix B.1 shows they use resolution 224 x 224 which is divisible by their patch sizes of 16 and 32. We will do the same.
- Implementing things in the order of the paper's method section isn't possible. Often, earlier steps mentioned, like learned positional embedding, require something later on, in this case an MLP.
- Adam optimizer actually can optimizer with respect to a weight matrix w that has the bias concatenated so we can write less code.
- Lots of minor design decisions, e.g. type of optimizers, ways to allocate memory in functions, dimensions to store weights matrix in linear layer ([out, in] better than [in, out] for the backward pass because of some CUDA and caching reasons).
- Learned about ABC and unittest libraries in this.
- Understanding softmax + CEloss pairing and going through the derivation of it to find loss with respect to logits. (Small implementation details like integer vs one-hot encoding.) Numerical stability issues with softmax (subtract max). Clipping probabilities to avoid log(0).
- Small import issues like using sys(..) only works if it's in a package... `__init__` needed
- Linear unit test tested forward and backward pass only for one pass. Needed to pass twice to check the `update_params()` function!
- Numerical issues with LN. Add eps in the bottom.
- Issue of using float64 in PyTorch vs float32 in cp for faster computation.
- Question of how to initialize positional embeddings? Decided on normal distribution as each patch needs to learn to move its embedding toward a certain ideal so should cluster around mean.
- Issue of no positional embedding PyTorch implementation... just have to be careful. Created some unit tests for shapes.
- Initial implementation of linear layer used the formula z = x @ W.T + b. Switched it back to z = W @ x + b because dimensions weren't working in MHA.
- ViTBlock backward pass commment. Important to talk about!
- Exploding gradients / vanishing
- cp.newaxis issues
- optimizing z = x @ W.T + b
- stacking CLS token/how to process
- not calculating softmax before classification
- running times weren't consistent with the same configuration
