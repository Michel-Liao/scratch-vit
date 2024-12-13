import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split


def download_and_preprocess_mnist(output_dir: str, val_split: float = 0.1) -> None:
    """
    Download and preprocess MNIST dataset.

    Args:
        output_dir: Directory to save processed files
        val_split: Fraction of training data to use for validation
    """
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
        ]
    )

    train_dataset = datasets.MNIST(
        root=output_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=output_dir, train=False, download=True, transform=transform
    )

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    def process_dataset(dataset, output_file):
        images = []
        labels = []

        for img, label in dataset:
            images.append(img.numpy()) # Will be (1, 28, 28) for MNIST
            labels.append(label)

        images = np.stack(images)  # Shape: (N, 1, 28, 28)
        labels = np.array(labels)

        # One-hot encode labels
        one_hot = np.zeros((labels.size, 10))
        one_hot[np.arange(labels.size), labels] = 1

        with open(output_file, "wb") as f:
            np.save(f, images)
            np.save(f, one_hot)

    process_dataset(train_dataset, os.path.join(output_dir, "mnist_train.npy"))
    process_dataset(val_dataset, os.path.join(output_dir, "mnist_val.npy"))
    process_dataset(test_dataset, os.path.join(output_dir, "mnist_test.npy"))


def download_and_preprocess_cifar10(output_dir: str, val_split: float = 0.1) -> None:
    """
    Download and preprocess CIFAR-10 dataset.

    Args:
        output_dir: Directory to save processed files
        val_split: Fraction of training data to use for validation
    """
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=output_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=output_dir, train=False, download=True, transform=transform
    )

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    def process_dataset(dataset, output_file):
        images = []
        labels = []

        for img, label in dataset:
            images.append(img.numpy())  # Will be (3, 32, 32) for CIFAR-10
            labels.append(label)

        images = np.stack(images)  # Shape: (N, 3, 32, 32)
        labels = np.array(labels)

        # One-hot encode labels (CIFAR-10 has 10 classes)
        one_hot = np.zeros((labels.size, 10))
        one_hot[np.arange(labels.size), labels] = 1

        with open(output_file, "wb") as f:
            np.save(f, images)
            np.save(f, one_hot)

    process_dataset(train_dataset, os.path.join(output_dir, "cifar10_train.npy"))
    process_dataset(val_dataset, os.path.join(output_dir, "cifar10_val.npy"))
    process_dataset(test_dataset, os.path.join(output_dir, "cifar10_test.npy"))


if __name__ == "__main__":
    output_dir = "./data"

    download_and_preprocess_mnist(f"{output_dir}/mnist")
    download_and_preprocess_cifar10(f"{output_dir}/cifar10")
    print(f"MNIST and CIFAR-10 datasets processed and saved to {output_dir}")
