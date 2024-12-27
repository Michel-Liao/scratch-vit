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
├── data/           (Datasets stored here)
├── scripts/        (Example batch script for random hyperparameter search)
├── src/            (Source code for from-scratch implementation)
├── tests/          (Unit tests to verify against PyTorch)
├── visualize/      (Example visualization scripts/notebooks)
├── preprocess.py   (Python script to prepare data)
├── trainer.py      (Train script houses the ViT trainer class)
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

### Training the Vision Transformer
The Vision Transformer can be trained using the script with various command-line arguments to customize the training process:

#### Data and Training Parameters
- `--data_path`: (Required) Path to the dataset folder
- `--mnist`: Flag to use MNIST dataset; if not set, CIFAR-10 will be used (default: True)
- `--num_patches`: (Required) Number of patches per row
- `--batch_size`: Number of samples per training batch (default: 16)
- `--epochs`: Total number of training epochs (default: 5)
- `--eval_interval`: Test evaluation interval (default: 2)

#### Model Architecture Parameters
- `--num_classes`: Number of classes in the dataset (default: 10)
- `--hidden_dim`: Dimensionality of the hidden layers (default: 128)
- `--num_heads`: Number of attention heads in multi-head attention (default: 4)
- `--num_blocks`: Number of transformer encoder blocks (default: 4)

#### Optimization Parameters
- `--learning_rate`: Learning rate for optimization (default: 1e-9)
- `--init_method`: Weight initialization method, choose from ["he", "normal", "uniform", "xavier"] (default: "he")

#### Example Usage
### Example Usage

Basic training on MNIST with default parameters:
```python
python train.py --data_path /path/to/mnist --num_patches 7
```

Training on CIFAR-10 with custom parameters:
```
python train.py \
    --data_path /path/to/cifar10 \
    --mnist False \
    --num_patches 8 \
    --batch_size 32 \
    --epochs 10 \
    --hidden_dim 256 \
    --num_heads 8 \
    --num_blocks 6 \
    --learning_rate 1e-4 \
    --init_method xavier
```

### Unit Tests

To run the unit tests:

1. `cd tests`
2. `python -m unittest [scriptname].py`

## Citations

```
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, A},
  journal={Advances in Neural Information Processing Systems},
  year={2017}
}

@article{kingma2014adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P},
  journal={arXiv preprint arXiv:1412.6980},
  year={2014}
}
@inproceedings{he2015delving,
  title={Delving deep into rectifiers: Surpassing human-level performance on imagenet classification},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1026--1034},
  year={2015}
}
@inproceedings{glorot2010understanding,
  title={Understanding the difficulty of training deep feedforward neural networks},
  author={Glorot, Xavier and Bengio, Yoshua},
  booktitle={Proceedings of the thirteenth international conference on artificial intelligence and statistics},
  pages={249--256},
  year={2010},
  organization={JMLR Workshop and Conference Proceedings}
}
@article{hendrycks2016gaussian,
  title={Gaussian error linear units (gelus)},
  author={Hendrycks, Dan and Gimpel, Kevin},
  journal={arXiv preprint arXiv:1606.08415},
  year={2016}
}
@article{agarap2018deep,
  title={Deep learning using rectified linear units},
  author={Agarap, Abien Fred},
  journal={arXiv preprint arXiv:1803.08375},
  year={2018}
}
@article{lei2016layer,
  title={Layer normalization},
  author={Lei Ba, Jimmy and Kiros, Jamie Ryan and Hinton, Geoffrey E},
  journal={ArXiv e-prints},
  pages={arXiv--1607},
  year={2016}
}
```

## Notes

### Patchify and Image Dimensions
One significant issue arose during the patchification process. If the image dimensions are not divisible by the chosen patch size, handling the remainder becomes problematic. The Vision Transformer (ViT) paper does not explicitly address this case, although Appendix B.1 mentions using a resolution of 224 × 224, which is divisible by the patch sizes of 16 and 32. To resolve this, we followed a similar approach, ensuring our input images were resized to compatible dimensions.

### Implementation Order and Dependencies
Implementing components in the order presented in the ViT paper's methodology was often impractical due to interdependencies. For example, implementing the learned positional embedding required an MLP that was described in a later section of the paper. This necessitated a non-linear development workflow to align the implementation with functional dependencies.

### Optimizations and Design Decisions
Various design decisions posed challenges, such as selecting appropriate optimizers and determining memory allocation strategies. We optimized the weight matrix in linear layers using the layout `[out, in]` instead of `[in, out]` to improve performance in backward passes due to CUDA caching. Additionally, we discovered that the Adam optimizer could optimize weights and biases concatenated into a single matrix, simplifying code.

### Numerical Stability and Loss Function Challenges
Understanding and implementing the pairing of softmax with cross-entropy loss (CE loss) required a deep dive into their derivations. Numerical stability issues with softmax necessitated subtracting the maximum value before exponentiation. Similarly, clipping probabilities was crucial to avoid undefined logarithms (e.g., log(0)). For layer normalization, adding a small ε value to the denominator addressed numerical instability.

### Unit Testing and Debugging
Unit testing revealed critical implementation bugs. For instance, our initial test of the linear layer's forward and backward passes did not account for testing parameter updates across multiple passes. This oversight was identified when testing the `update_params()` function. Furthermore, the absence of built-in PyTorch positional embedding required custom implementation and additional unit tests to verify shape correctness.

### Implementation-Specific Challenges
Several implementation-specific issues arose, including:

* Import errors when not using a proper package structure, resolved by adding `__init__.py`
* Deciding how to initialize positional embeddings; we opted for a normal distribution to allow patches to cluster around a mean and learn ideal embeddings
* Addressing discrepancies between PyTorch's `float32` and CuPy's `float64`, which impacted computation speed and memory usage
* Handling `cp.newaxis` and optimizing matrix operations like `z = x @ W.T + b`

### Model-Specific Issues
Key model-specific issues included:

* Designing and testing the backward pass in the ViT block, particularly for multi-head attention (MHA)
* Stacking the CLS token and processing it efficiently during training and evaluation
* Managing exploding and vanishing gradients, which required careful tuning of the learning rate and gradient clipping
* Ensuring the correct sequence of operations, such as not applying softmax prematurely before the classification layer
* Identifying and fixing issues with MNIST one-hot encoding that were causing incorrect label representations
