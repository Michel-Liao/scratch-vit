import argparse
import os
from typing import Tuple
import pickle

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from model.loss import CategoricalCrossEntropyLoss
from model.optimizers import Adam
from model.vit_finished import ViT
from model.softmax import Softmax


class VisionTransformer:
    """Vision Transformer (ViT) training wrapper for CIFAR-10."""

    def __init__(
        self,
        path_to_cifar: str,
        batch_size: int,
        epochs: int,
        test_epoch_interval: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        learning_rate: float,
        patch_size: int,
        init_method: str,
    ) -> None:
        """Initialize Vision Transformer trainer.

        Args:
            path_to_cifar: Path to folder containing CIFAR-10 dataset files
            batch_size: Number of samples per training batch
            epochs: Number of training epochs
            test_epoch_interval: Interval for running test evaluation
            hidden_dim: Hidden dimension size for transformer
            num_heads: Number of attention heads
            num_blocks: Number of transformer blocks
            learning_rate: Learning rate for optimizer
            patch_size: Size of image patches (assumes square patches)
            init_method: Type of linear layer initialization ('he', 'normal', 'uniform', 'xavier')
        """
        self.path_to_cifar = path_to_cifar
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_epoch_interval = test_epoch_interval
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.init_method = init_method

        # CIFAR-10 class names
        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load and prepare CIFAR-10 dataset."""
        self.x_train, self.y_train = self._load_and_process_training_data()
        self.x_test, self.y_test = self._load_and_process_test_data()

    def _unpickle(self, file):
        """Helper function to unpickle CIFAR-10 data files."""
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    def _load_and_process_training_data(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """Load and process CIFAR-10 training data.

        Returns:
            Tuple containing processed features and one-hot encoded labels
        """
        x_train = []
        y_train = []

        # CIFAR-10 training data is split into 5 batches
        for i in range(1, 6):
            filename = os.path.join(self.path_to_cifar, f"data_batch_{i}")
            batch_dict = self._unpickle(filename)

            # Extract data and labels
            x_batch = batch_dict[b"data"]
            y_batch = batch_dict[b"labels"]

            x_train.append(x_batch)
            y_train.extend(y_batch)

        # Concatenate all training batches
        x_train = cp.asarray(cp.vstack(x_train), dtype=cp.float32)
        y_train = cp.asarray(y_train, dtype=cp.int32)

        # Reshape and normalize images
        x_train = x_train.reshape(-1, 3, 32, 32)
        x_train = x_train / 255.0  # Normalize to [0, 1]

        # One-hot encode labels
        num_samples = len(y_train)
        y_onehot = cp.zeros((num_samples, 10), dtype=cp.float32)
        y_onehot[cp.arange(num_samples), y_train] = 1

        return x_train, y_onehot

    def _load_and_process_test_data(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """Load and process CIFAR-10 test data.

        Returns:
            Tuple containing processed features and one-hot encoded labels
        """
        test_file = os.path.join(self.path_to_cifar, "test_batch")
        test_dict = self._unpickle(test_file)

        x_test = cp.asarray(test_dict[b"data"], dtype=cp.float32)
        y_test = cp.asarray(test_dict[b"labels"], dtype=cp.int32)

        # Reshape and normalize images
        x_test = x_test.reshape(-1, 3, 32, 32)
        x_test = x_test / 255.0  # Normalize to [0, 1]

        # One-hot encode labels
        num_samples = len(y_test)
        y_onehot = cp.zeros((num_samples, 10), dtype=cp.float32)
        y_onehot[cp.arange(num_samples), y_test] = 1

        return x_test, y_onehot

    def datafeeder(self, x: cp.ndarray, y: cp.ndarray, shuffle: bool = False):
        """Generate batches of data.

        Args:
            x: Input images (N, C, H, W)
            y: Labels (N, L)
            shuffle: Whether to shuffle data

        Yields:
            Tuple of (batch_images, batch_labels)
        """
        n_samples = len(y)

        if shuffle:
            randomize = cp.arange(n_samples)
            cp.random.shuffle(randomize)
            x = x[randomize]
            y = y[randomize]

        for i in range(0, n_samples, self.batch_size):
            batch_end = min(i + self.batch_size, n_samples)
            yield x[i:batch_end], y[i:batch_end]

    def train_iter(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss for the epoch
        """
        dataloader = self.datafeeder(self.x_train, self.y_train, shuffle=True)
        train_losses = []
        total_batches = len(self.y_train) // self.batch_size

        for batch in tqdm.tqdm(dataloader, total=total_batches, desc="Training"):
            x, y = batch
            y_pred = self.model.forward(x)
            loss = self.loss_function.forward(y_pred, y)
            error = self.loss_function.backward(y)
            self.model.backward(error)
            self.model.update_params()
            train_losses.append(loss)

        return float(cp.mean(cp.asarray(train_losses)))

    def test_iter(self) -> Tuple[float, float]:
        """Evaluate model on test set.

        Returns:
            Tuple of (average test loss, test accuracy)
        """
        test_dataloader = self.datafeeder(self.x_test, self.y_test)
        test_losses = []
        correct_predictions = 0
        total_samples = 0
        total_batches = len(self.y_test) // self.batch_size

        for batch in tqdm.tqdm(test_dataloader, total=total_batches, desc="Testing"):
            x, y = batch
            y_pred = self.model.forward(x)
            loss = self.loss_function.forward(y_pred, y)

            # Calculate accuracy
            y_prob = Softmax()(y_pred)
            y_pred_class = cp.argmax(y_prob, axis=-1, keepdims=True)
            y_true_class = cp.argmax(y, axis=-1, keepdims=True)
            correct_predictions += int(cp.sum(y_pred_class == y_true_class))
            total_samples += y.shape[0]

            test_losses.append(loss)

        avg_loss = float(cp.mean(cp.asarray(test_losses)))
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def compute_confusion_matrix(self, x: cp.ndarray, y: cp.ndarray) -> np.ndarray:
        """Compute confusion matrix for the given dataset.

        Args:
            x: Input images (N, C, H, W)
            y: One-hot encoded labels (N, num_classes)

        Returns:
            confusion_matrix: (num_classes, num_classes) numpy array
        """
        confusion_matrix = np.zeros((10, 10), dtype=np.int32)
        dataloader = self.datafeeder(x, y)
        total_batches = len(y) // self.batch_size

        for batch in tqdm.tqdm(
            dataloader, total=total_batches, desc="Computing confusion matrix"
        ):
            batch_x, batch_y = batch
            y_pred = self.model.forward(batch_x)
            y_prob = Softmax()(y_pred)

            # Convert predictions and true labels to class indices
            y_pred_class = cp.argmax(y_prob, axis=-1)
            y_true_class = cp.argmax(batch_y, axis=-1)

            # Update confusion matrix
            y_pred_cpu = cp.asnumpy(y_pred_class)
            y_true_cpu = cp.asnumpy(y_true_class)

            for true_label, pred_label in zip(y_true_cpu, y_pred_cpu):
                confusion_matrix[true_label, pred_label] += 1

        return confusion_matrix

    def plot_confusion_matrix(
        self, confusion_matrix: np.ndarray, title: str, save_path: str
    ):
        """Plot and save confusion matrix heatmap.

        Args:
            confusion_matrix: Confusion matrix array
            title: Title for the plot
            save_path: Path to save the confusion matrix plot
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def train_model(self) -> None:
        """Train the Vision Transformer model and compute confusion matrices."""
        # Initialize model with CIFAR-10 dimensions
        self.model = ViT(
            im_dim=(3, 32, 32),
            n_patches=self.patch_size,
            h_dim=self.hidden_dim,
            n_heads=self.num_heads,
            num_blocks=self.num_blocks,
            classes=10,
            init_method=self.init_method,
        )

        self.loss_function = CategoricalCrossEntropyLoss()
        self.optimizer = Adam(lr=self.learning_rate)
        self.model.init_optimizer(self.optimizer)

        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            train_loss = self.train_iter()
            print(f"Training Loss: {train_loss:.4f}")

            if (epoch + 1) % self.test_epoch_interval == 0:
                test_loss, test_acc = self.test_iter()
                print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

            # Compute confusion matrices in the last epoch
            if epoch == self.epochs - 1:
                print("\nComputing final confusion matrices...")

                # Training set confusion matrix
                train_conf_matrix = self.compute_confusion_matrix(
                    self.x_train, self.y_train
                )
                self.plot_confusion_matrix(
                    train_conf_matrix,
                    f"Training Set Confusion Matrix (Epoch {epoch + 1})",
                    "train_confusion_matrix_cifar.png",
                )

                # Test set confusion matrix
                test_conf_matrix = self.compute_confusion_matrix(
                    self.x_test, self.y_test
                )
                self.plot_confusion_matrix(
                    test_conf_matrix,
                    f"Test Set Confusion Matrix (Epoch {epoch + 1})",
                    "test_confusion_matrix_cifar.png",
                )

                print(
                    "Confusion matrices have been saved as 'train_confusion_matrix.png' and 'test_confusion_matrix.png'"
                )

    # def train_model(self) -> None:
    #     """Train the Vision Transformer model."""
    #     # Initialize model with CIFAR-10 dimensions
    #     self.model = ViT(
    #         im_dim=(3, 32, 32),  # CIFAR-10 images are 32x32x3
    #         n_patches=self.patch_size,
    #         h_dim=self.hidden_dim,
    #         n_heads=self.num_heads,
    #         num_blocks=self.num_blocks,
    #         classes=10,  # CIFAR-10 has 10 classes
    #         init_method=self.init_method,
    #     )

    #     self.loss_function = CategoricalCrossEntropyLoss()
    #     self.optimizer = Adam(lr=self.learning_rate)
    #     self.model.init_optimizer(self.optimizer)

    #     print(f"Starting training for {self.epochs} epochs...")
    #     for epoch in range(self.epochs):
    #         print(f"\nEpoch {epoch + 1}/{self.epochs}")
    #         train_loss = self.train_iter()
    #         print(f"Training Loss: {train_loss:.4f}")

    #         if (epoch + 1) % self.test_epoch_interval == 0:
    #             test_loss, test_acc = self.test_iter()
    #             print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train Vision Transformer on CIFAR-10")

    # Data and training parameters
    parser.add_argument(
        "--path_to_cifar", required=True, help="Path to CIFAR-10 dataset folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--test_epoch_interval", type=int, default=1, help="Test evaluation interval"
    )

    # Model architecture parameters
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=6, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--patch_size", type=int, default=8, help="Size of image patches"
    )

    # Optimization parameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--init_method",
        choices=["he", "normal", "uniform", "xavier"],
        default="he",
        help="Linear layer initialization type",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vit_cifar = VisionTransformer(
        path_to_cifar=args.path_to_cifar,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_epoch_interval=args.test_epoch_interval,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        learning_rate=args.learning_rate,
        patch_size=args.patch_size,
        init_method=args.init_method,
    )

    vit_cifar.train_model()
