import argparse
from typing import Tuple, Generator

import cupy as cp
from tqdm import tqdm

from src.loss import CategoricalCrossEntropyLoss
from src.optimizers import Adam
from src.ViT import ViT
from src.softmax import Softmax


class TrainerViT:
    """
    Training wrapper for Vision Transformer (ViT)
    """

    def __init__(
        self,
        data_path: str,
        classes: list,
        batch_size: int,
        epochs: int,
        eval_interval: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        learning_rate: float,
        patch_size: int,
        init_method: str,
    ) -> None:
        """
        Initialize the Vision Transformer trainer.

        Args:
            data_path_train (str): Path to the dataset train, val, and test files.
            classes (list): List of class names or labels for the classification task.
            batch_size (int): Number of samples per batch during training.
            epochs (int): Total number of training epochs.
            eval_interval (int): Number of epochs between evaluations of the model on the validation set.
            hidden_dim (int): Dimensionality of the hidden layers in the Vision Transformer architecture.
            num_heads (int): Number of attention heads in the multi-head self-attention mechanism.
            num_blocks (int): Number of Transformer encoder blocks in the model.
            learning_rate (float): Initial learning rate for the optimizer.
            patch_size (int): Size of image patches to be fed into the Transformer.
            init_method (str): Initialization method for model weights (e.g., "xavier", "he", "normal").
        """

        self.data_path = data_path
        self.classes = classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.init_method = init_method

        self.model = ViT(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            patch_size=self.patch_size,
            num_classes=len(self.classes),
            init_method=self.init_method,
        )

        self.loss_function = CategoricalCrossEntropyLoss()
        self.model.init_optimizer(Adam(lr=self.learning_rate))

        self.load_dataset()

    def load_dataset(self) -> None:
        """
        Load preprocessed datasets directly from the provided paths.
        Assumes data is already in the correct format. One-hot encodes the class labels.
        """
        with open(f"{self.data_path}_train", "rb") as f:
            self.x_train = cp.load(f)
            self.y_train = cp.load(f)

        with open(f"{self.data_path}_val", "rb") as f:
            self.x_val = cp.load(f)
            self.y_val = cp.load(f)

        with open(f"{self.data_path}_test", "rb") as f:
            self.x_test = cp.load(f)
            self.y_test = cp.load(f)

    def dataloader(
        self, X: cp.ndarray, Y: cp.ndarray, shuffle: bool = False
    ) -> Generator[Tuple[cp.ndarray, cp.ndarray], None, None]:
        """
        Data generator for training and evaluation.

        Args:
            X (cp.ndarray): Input images (n, C, H, W).
            Y (cp.ndarray): One-hot encoded target labels (n, L).
            shuffle (bool): Whether to shuffle the data before feeding it.

        Returns:
            Tuple of (X_batch, Y_batch): The next batch of data.
        """

        assert X.shape[0] == Y.shape[0], "Input and target dimensions do not match."

        num_samples = X.shape[0]

        if shuffle:
            randomize = cp.arange(num_samples)
            cp.random.shuffle(randomize)
            X = X[randomize, :, :, :]
            Y = Y[randomize, :]

        for batch_start in range(0, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)

            X_batch = X[batch_start:batch_end, :, :, :]
            Y_batch = Y[batch_start:batch_end, :]

            yield X_batch, Y_batch

    def train_iter(self) -> float:
        """
        Training the model for one epoch.

        Returns:
            float: Average training loss for the epoch.
        """

        loader = self.dataloader(self.x_train, self.y_train, shuffle=True)
        train_losses = []
        num_batches = len(self.y_train) // self.batch_size

        for X_batch, Y_batch in tqdm(loader, total=num_batches, desc="Training"):
            Y_logits = self.model.forward(X_batch)
            loss = self.loss_function.forward(Y_logits, Y_batch)
            train_losses.append(loss)

            grad = self.loss_function.backward(Y_batch)
            self.model.backward(grad)
            self.model.update_params()

        return sum(train_losses) / len(train_losses)

    def evaluate(self, validation: bool = False) -> Tuple[float, float]:
        """
        Evaluate the model on the validation set.

        Args:
            validation (bool): Whether to evaluate on the validation set or the test set.

        Returns:
            Tuple[float, float]: Average validation loss and accuracy.
        """

        if validation:
            loader = self.dataloader(self.x_val, self.y_val)
            num_samples = len(self.y_val)
        else:
            loader = self.dataloader(self.x_test, self.y_test)
            num_samples = len(self.y_test)

        losses = []
        correct = 0

        for X_batch, Y_batch in tqdm(loader, total=num_samples, desc="Validation"):
            Y_logits = self.model.forward(X_batch)
            loss = self.loss_function.forward(Y_pred, Y_batch)
            losses.append(loss)

            Y_probabilities = Softmax()(Y_logits)
            Y_pred = cp.argmax(Y_probabilities, axis=1, keepdims=True)
            Y_true = cp.argmax(Y_batch, axis=1, keepdims=True)

            correct += int(cp.sum(Y_pred == Y_true))

        return sum(losses) / len(losses), correct / num_samples

    def train(self) -> None:
        """
        Train the Vision Transformer model.
        """

        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            train_loss = self.train_iter()
            print(f"Training Loss: {train_loss:.4f}")

            if (epoch + 1) % self.test_epoch_interval == 0:
                val_loss, val_acc = self.evaluate(validation=True)
                print(
                    f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
                )

    def test(self) -> None:
        """
        Test the Vision Transformer model on the test set.
        """

        test_loss, test_acc = self.evaluate(validation=False)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train Vision Transformer on MNIST")

    # Data and training parameters
    parser.add_argument("--data_path", required=True, help="Path to dataset folder")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=2, help="Test evaluation interval"
    )

    # Model architecture parameters
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of classes in the dataset"
    )
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainer = TrainerViT(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        learning_rate=args.learning_rate,
        patch_size=args.patch_size,
        init_method=args.init_method,
    )

    trainer.train()
    trainer.test()
