import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import seaborn as sns
import tqdm
from model.loss import CategoricalCrossEntropyLoss
from model.optimizers import Adam
from model.vit_finished import ViT
from model.softmax import Softmax


class VisionTransformer:
    """Vision Transformer (ViT) training wrapper for MNIST."""

    def __init__(
        self,
        path_to_mnist: str,
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
            path_to_mnist: Path to folder containing MNIST dataset files
            batch_size: Number of samples per training batch
            epochs: Number of training epochs
            test_epoch_interval: Interval for running test evaluation
            hidden_dim: Hidden dimension size for transformer
            num_heads: Number of attention heads
            num_blocks: Number of transformer blocks
            learning_rate: Learning rate for optimizer
            patch_size: Size of image patches (assumes square patches)
            linear_init: Type of linear layer initialization ('normal', 'uniform', 'xavier')
        """
        self.path_to_mnist = path_to_mnist
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_epoch_interval = test_epoch_interval
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.init_method = init_method

        self.load_dataset()

    def load_dataset(self) -> None:
        """Load and prepare MNIST dataset."""
        self.x_train, self.y_train = self._load_and_process_data("mnist_train.npy")
        self.x_test, self.y_test = self._load_and_process_data("mnist_test.npy")

    def _load_and_process_data(self, filename: str) -> Tuple[cp.ndarray, cp.ndarray]:
        """Load and process MNIST data file.

        Args:
            filename: Name of the MNIST data file

        Returns:
            Tuple containing processed features and one-hot encoded labels
        """
        with open(os.path.join(self.path_to_mnist, filename), "rb") as f:
            x_data = cp.load(f)
            # # Cut x_data in half
            # x_data = x_data[:, : x_data.shape[1] // 2]
            y_data = cp.load(f).astype(cp.int32)
            # y_data = y_data[: y_data.size // 2]

        num_samples = int(y_data.size)
        num_classes = int(y_data.max()) + 1

        # One-hot encode labels
        y_onehot = cp.zeros((num_samples, num_classes), dtype=cp.float32)
        y_onehot[cp.arange(num_samples), y_data] = 1

        return x_data, y_onehot

    def datafeeder(self, x: cp.ndarray, y: cp.ndarray, shuffle: bool = False):
        """Generate batches of data.

        Args:
            x: Input images (C, N, H, W)
            y: Labels (N, L)
            shuffle: Whether to shuffle data

        Yields:
            Tuple of (batch_images, batch_labels)
        """
        n_samples = len(y)

        if shuffle:
            randomize = cp.arange(n_samples)
            cp.random.shuffle(randomize)
            x = x[:, randomize]
            y = y[randomize]

        for i in range(0, n_samples, self.batch_size):
            batch_end = min(i + self.batch_size, n_samples)
            batch_size = batch_end - i

            x_batch = x[:, i:batch_end].transpose(1, 0)
            x_batch = x_batch.reshape(batch_size, 1, 28, 28)
            y_batch = y[i:batch_end]

            yield x_batch, y_batch

    def train_iter(self, collect_predictions: bool = False) -> Tuple[float, list, list]:
        """Run one training epoch.

        Args:
            collect_predictions: Whether to collect predictions and true labels

        Returns:
            Tuple of (average loss, predictions list, true labels list)
        """
        dataloader = self.datafeeder(self.x_train, self.y_train, shuffle=True)
        train_losses = []
        total_batches = len(self.y_train) // self.batch_size

        all_predictions = []
        all_true_labels = []

        for batch in tqdm.tqdm(dataloader, total=total_batches, desc="Training"):
            x, y = batch
            y_pred = self.model.forward(x)
            loss = self.loss_function.forward(y_pred, y)
            error = self.loss_function.backward(y)
            self.model.backward(error)
            self.model.update_params()
            train_losses.append(loss)

            if collect_predictions:
                # Convert predictions and labels to class indices
                y_prob = Softmax()(y_pred)
                pred_classes = cp.argmax(y_prob, axis=-1)
                true_classes = cp.argmax(y, axis=-1)

                # Move to CPU for storage
                all_predictions.extend(cp.asnumpy(pred_classes))
                all_true_labels.extend(cp.asnumpy(true_classes))

        avg_loss = float(cp.mean(cp.asarray(train_losses)))
        return avg_loss, all_predictions, all_true_labels

    # def train_iter(self) -> float:
    #     """Run one training epoch.

    #     Returns:
    #         Average training loss for the epoch
    #     """
    #     dataloader = self.datafeeder(self.x_train, self.y_train, shuffle=True)
    #     train_losses = []
    #     total_batches = len(self.y_train) // self.batch_size

    #     for batch in tqdm.tqdm(dataloader, total=total_batches, desc="Training"):
    #         x, y = batch
    #         y_pred = self.model.forward(x)
    #         loss = self.loss_function.forward(y_pred, y)
    #         error = self.loss_function.backward(y)
    #         self.model.backward(error)
    #         self.model.update_params()
    #         train_losses.append(loss)

    #     return float(cp.mean(cp.asarray(train_losses)))

    def compute_confusion_matrix(
        self, y_true: list, y_pred: list, num_classes: int = 10
    ) -> cp.ndarray:
        """Compute confusion matrix from predictions and true labels.

        Args:
            y_true: List of true class labels
            y_pred: List of predicted class labels
            num_classes: Number of classes in the dataset

        Returns:
            Confusion matrix as a numpy array
        """
        conf_matrix = cp.zeros((num_classes, num_classes), dtype=cp.int32)
        for t, p in zip(y_true, y_pred):
            conf_matrix[t, p] += 1
        return conf_matrix

    def plot_confusion_matrix(
        self, confusion_matrix: cp.ndarray, save_path: str = None
    ) -> None:
        """Plot confusion matrix using seaborn's heatmap.

        Args:
            confusion_matrix: Computed confusion matrix
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))

        # Convert confusion matrix to numpy and normalize
        cm = cp.asnumpy(confusion_matrix)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=cm,  # Show raw counts in cells
            fmt="d",  # Format annotations as integers
            cmap="Blues",
            xticklabels=range(10),
            yticklabels=range(10),
        )

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Confusion matrix plot saved to {save_path}")

        plt.show()
        plt.close()

    def train_model(self, confusion_matrix_path: str = None) -> None:
        """Train the Vision Transformer model.

        Args:
            confusion_matrix_path: Optional path to save confusion matrix plot
        """
        # Initialize model
        self.model = ViT(
            im_dim=(1, 28, 28),
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

            # Collect predictions only on the final epoch
            collect_predictions = epoch == self.epochs - 1
            train_loss, predictions, true_labels = self.train_iter(collect_predictions)
            print(f"Training Loss: {train_loss:.4f}")

            if (epoch + 1) % self.test_epoch_interval == 0:
                test_loss, test_acc = self.test_iter()
                print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

            # Compute and display confusion matrix on the final epoch
            if collect_predictions:
                conf_matrix = self.compute_confusion_matrix(true_labels, predictions)
                print("\nConfusion Matrix:")
                print(cp.asnumpy(conf_matrix))

                # Plot and optionally save the confusion matrix
                self.plot_confusion_matrix(conf_matrix, confusion_matrix_path)

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train Vision Transformer on MNIST")

    # Data and training parameters
    parser.add_argument(
        "--path_to_mnist", required=True, help="Path to MNIST dataset folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--test_epoch_interval", type=int, default=2, help="Test evaluation interval"
    )

    # Model architecture parameters
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=4, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--patch_size", type=int, default=7, help="Size of image patches"
    )

    # Optimization parameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-9, help="Learning rate"
    )
    parser.add_argument(
        "--init_method",
        choices=["he", "normal", "uniform", "xavier"],
        default="he",
        help="Linear layer initialization type",
    )

    parser.add_argument(
        "--confusion_matrix_path", type=str, help="Path to save confusion matrix plot"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vit_mnist = VisionTransformer(
        path_to_mnist=args.path_to_mnist,
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

    vit_mnist.train_model(args.confusion_matrix_path)
