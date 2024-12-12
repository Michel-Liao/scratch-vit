import argparse
import os
import json
from datetime import datetime
from pathlib import Path

from model.loss import CategoricalCrossEntropyLoss
from model.optimizers import Adam
from model.vit_finished import ViT
from model.softmax import Softmax
import cupy as cp
import tqdm


class VisionTransformer:
    """VIT implementation Wrapper with enhanced hyperparameter support."""

    def __init__(
        self,
        path_to_mnist: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        hidden_dim: int,
        patch_size: int,
        num_heads: int,
        num_blocks: int,
        test_epoch_interval: int,
        output_dir: str,
    ) -> None:
        """Initialize.

        Args:
            path_to_mnist: Path to folder containing MNIST data
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            hidden_dim: Hidden dimension size
            patch_size: Size of image patches
            num_heads: Number of attention heads
            num_blocks: Number of transformer blocks
            test_epoch_interval: Interval for running test evaluation
            output_dir: Directory to save model outputs
        """
        self.path_to_mnist = path_to_mnist
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.test_epoch_interval = test_epoch_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.load_dataset_from_file(path_to_mnist)
        self.best_test_acc = 0.0
        self.training_history = {
            "train_loss": [],
            "test_loss": [],
            "test_acc": [],
            "hyperparameters": self.get_hyperparameters(),
        }

    def get_hyperparameters(self) -> dict:
        """Get current hyperparameter configuration."""
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "patch_size": self.patch_size,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
        }

    # [Previous datafeeder and load_dataset_from_file methods remain unchanged]

    def train_iter(self) -> float:
        """Train model for one epoch.

        Returns:
            float: Average training loss for the epoch
        """
        dataloader = self.datafeeder(self.x_train, self.y_train, True)
        train_error = []
        total_len = len(self.y_train) // self.batch_size
        for batch in tqdm.tqdm(dataloader, total=total_len, desc="Training"):
            x, y = batch
            x = x.transpose(1, 0)
            x = x.reshape(self.batch_size, 1, 28, 28)
            y_hat = self.model.forward(x)
            loss = self.loss_function.forward(y_hat, y)
            error = self.loss_function.backward(y)
            self.model.backward(error)
            self.model.update_params()
            train_error.append(loss)
        avg_loss = float(cp.mean(cp.asarray(train_error)))
        return avg_loss

    def test_iter(self) -> tuple[float, float]:
        """Test model.

        Returns:
            tuple[float, float]: Average test loss and accuracy
        """
        test_dataloader = self.datafeeder(self.x_test, self.y_test)
        test_error = []
        epoch_tp = 0
        epoch_total = 0
        total_len = len(self.y_test) // self.batch_size
        for batch in tqdm.tqdm(test_dataloader, total=total_len, desc="Testing"):
            x, y = batch
            x = x.transpose(1, 0)
            x = x.reshape(self.batch_size, 1, 28, 28)
            y_hat = self.model.forward(x)
            loss = self.loss_function.forward(y_hat, y)
            y_hat = Softmax()(y_hat)
            y_pred = cp.argmax(y_hat, axis=-1)
            y_true = cp.argmax(y, axis=-1)
            correct = cp.sum(y_pred == y_true)
            total = cp.size(y_true)
            epoch_tp += correct
            epoch_total += total
            test_error.append(loss)

        avg_loss = float(cp.mean(cp.asarray(test_error)))
        accuracy = float(epoch_tp / epoch_total)
        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, test_acc: float) -> None:
        """Save model checkpoint if it's the best so far."""
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "test_acc": test_acc,
                "hyperparameters": self.get_hyperparameters(),
            }
            checkpoint_path = self.output_dir / f"best_model_acc{test_acc:.4f}.pt"
            cp.save(checkpoint_path, checkpoint)

    def save_training_history(self) -> None:
        """Save training history and hyperparameters."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=4)

    def train_model(self) -> None:
        """Train model with specified hyperparameters."""
        self.model = ViT(
            im_dim=(1, 28, 28),
            n_patches=self.patch_size,
            h_dim=self.hidden_dim,
            n_heads=self.num_heads,
            num_blocks=self.num_blocks,
            classes=10,
        )
        self.loss_function = CategoricalCrossEntropyLoss()
        self.optimizer = Adam(lr=self.learning_rate)
        self.model.init_optimizer(self.optimizer)

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            train_loss = self.train_iter()
            self.training_history["train_loss"].append(train_loss)

            if (epoch + 1) % self.test_epoch_interval == 0:
                test_loss, test_acc = self.test_iter()
                self.training_history["test_loss"].append(test_loss)
                self.training_history["test_acc"].append(test_acc)
                print(
                    f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                )
                self.save_checkpoint(epoch, test_acc)

        self.save_training_history()


def parse_args():
    """Parse command line arguments with enhanced hyperparameter support."""
    parser = argparse.ArgumentParser(description="Train Vision Transformer on MNIST")

    # Data and output arguments
    parser.add_argument("--path_to_mnist", required=True, help="Path to MNIST dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--test_epoch_interval", type=int, default=1, help="Test evaluation interval"
    )

    # Model architecture hyperparameters
    parser.add_argument(
        "--hidden_dim", type=int, default=8, help="Hidden dimension size"
    )
    parser.add_argument("--patch_size", type=int, default=7, help="Patch size")
    parser.add_argument(
        "--num_heads", type=int, default=2, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=2, help="Number of transformer blocks"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Create unique output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp

    vit_mnist = VisionTransformer(
        path_to_mnist=args.path_to_mnist,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        test_epoch_interval=args.test_epoch_interval,
        output_dir=output_dir,
    )
    vit_mnist.train_model()
