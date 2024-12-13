import argparse
import os

from src.loss import CategoricalCrossEntropyLoss as CrossEntropyLoss
from src.optimizers import Adam
from src.vit_finished import ViT
from src.softmax import Softmax
import cupy as cp
import tqdm


class VisionTransformer:
    """VIT implementation Wrapper."""

    def __init__(
        self, path_to_mnist: str, batch_size: int, epochs: int, test_epoch_interval: int
    ) -> None:
        """Initialize.

        Args:
            path_to_mnist: path to folder containing mnist.
            batch_size: batch size.
            epochs: number of epochs.
            test_epoch_interval: test epoch run interval.
        """
        self.path_to_mnist = path_to_mnist
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_epoch_interval = test_epoch_interval
        self.load_dataset_from_file(path_to_mnist)

    def datafeeder(self, x: cp.ndarray, y: cp.ndarray, shuffle: bool = False):
        """Datafeeder for train test.
        Args:
            x: input images in format (C, N, H, W) where N is number of samples.
            y: label in format (N, L) where N is number of samples, L is label dimension.
            shuffle: shuffle data.
        Yields:
            x: batch of images in format (B, C, H, W) where B is batch size
            y: batch of labels in format (B, L)
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

            # Get batch data
            x_batch = x[:, i:batch_end]  # (C, B, H, W)
            y_batch = y[i:batch_end]  # (B, L)

            # Reshape x to (B, C, H, W)
            x_batch = x_batch.transpose(1, 0)
            x_batch = x_batch.reshape(batch_size, 1, 28, 28)

            yield x_batch, y_batch

    def load_dataset_from_file(self, path_to_mnist: str) -> None:
        """Load dataset from file.

        Args:
            path_to_mnist: path to folder containing mnist.
        """
        with open(os.path.join(path_to_mnist, "mnist_train.npy"), "rb") as f:
            x_data = cp.load(f)  # Feature data
            y_data = cp.load(f)  # Labels

        # Ensure y_data contains integers
        y_data = y_data.astype(cp.int32)

        # Get the number of samples and classes
        num_samples = int(y_data.size)  # Total number of labels
        num_classes = int(y_data.max()) + 1  # Maximum label + 1 for one-hot

        # Initialize a zero matrix for one-hot encoding
        y_onehot = cp.zeros((num_samples, num_classes), dtype=cp.float32)

        # Perform one-hot encoding
        y_onehot[cp.arange(num_samples), y_data] = 1

        # Set the class attributes
        self.x_train = x_data
        self.y_train = y_onehot

        with open(os.path.join(path_to_mnist, "mnist_test.npy"), "rb") as f:
            x_data = cp.load(f)  # Feature data
            y_data = cp.load(f)  # Labels

        # Ensure y_data contains integers
        y_data = y_data.astype(cp.int32)

        # Get the number of samples and classes
        num_samples = int(y_data.size)  # Total number of labels
        num_classes = int(y_data.max()) + 1  # Maximum label + 1 for one-hot

        # Initialize a zero matrix for one-hot encoding
        y_onehot = cp.zeros((num_samples, num_classes), dtype=cp.float32)

        # Perform one-hot encoding
        y_onehot[cp.arange(num_samples), y_data] = 1

        # Set the class attributes
        self.x_test = x_data
        self.y_test = y_onehot

    def train_iter(self) -> None:
        """Train model for one epoch."""
        dataloader = self.datafeeder(self.x_train, self.y_train, True)
        train_error = []
        total_len = len(self.y_train) // self.batch_size
        for batch in tqdm.tqdm(dataloader, total=total_len):
            x, y = batch
            y_hat = self.model.forward(x)
            loss = self.loss_function.forward(y_hat, y)
            error = self.loss_function.backward(y)
            self.model.backward(error)
            self.model.update_params()
            train_error.append(loss)
        print(cp.mean(cp.asarray(train_error)))

    def test_iter(self) -> None:
        """Test model."""
        test_dataloader = self.datafeeder(self.x_test, self.y_test)
        test_error = []
        epoch_tp = 0
        epoch_total = 0
        total_len = len(self.y_test) // self.batch_size
        for batch in tqdm.tqdm(test_dataloader, total=total_len):
            x, y = batch
            y_hat = self.model.forward(x)
            loss = self.loss_function.forward(y_hat, y)
            y_hat = Softmax()(y_hat)
            y_pred = cp.argmax(y_hat, axis=-1, keepdims=True)
            correct = cp.sum(y_pred == y)
            total = cp.size(y)
            epoch_tp += correct
            epoch_total += total
            test_error.append(loss)
        print("test error", cp.mean(cp.asarray(test_error)))
        print("test acc", epoch_tp / epoch_total)

    def train_model(self) -> None:
        """Train model."""
        self.model = ViT(
            im_dim=(1, 28, 28),
            n_patches=7,
            h_dim=768,
            n_heads=12,
            num_blocks=12,
            classes=10,
        )
        self.loss_function = CrossEntropyLoss()
        self.optimizer = Adam(lr=1e-9)  # SGD()
        self.model.init_optimizer(self.optimizer)
        for epoch in range(self.epochs):
            self.train_iter()
            if (epoch + 1) % self.test_epoch_interval == 0:
                self.test_iter()


def parse_args():
    """Parse the arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_mnist",
        dest="path_to_mnist",
        required=True,
    )
    parser.add_argument("--batch_size", dest="batch_size", required=False, default=16)
    parser.add_argument("--epochs", dest="epochs", required=False, default=10)
    parser.add_argument(
        "--test_epoch_interval", dest="test_epoch_interval", required=False, default=2
    )
    args = parser.parse_args()
    return (args.path_to_mnist, args.batch_size, args.epochs, args.test_epoch_interval)


if __name__ == "__main__":
    path_to_mnist, batch_size, epochs, test_epoch_interval = parse_args()
    vit_mnist = VisionTransformer(
        path_to_mnist, batch_size, epochs, test_epoch_interval
    )
    vit_mnist.train_model()
