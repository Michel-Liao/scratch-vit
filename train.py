import argparse
import os

from model.loss import CategoricalCrossEntropyLoss as CrossEntropyLoss
from model.optimizers import Adam
from model.vit_finished import ViT
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
            x: input images.
            y: label.
            shuffle: shuffle data.

        Yields:
            a batch of data
        """
        if shuffle:
            randomize = cp.arange(len(y))
            cp.random.shuffle(randomize)
            x = x[:, randomize]
            y = y[randomize]
        for i in range(0, len(y), self.batch_size):
            yield x[:, i : i + self.batch_size], y[i : i + self.batch_size]

    def load_dataset_from_file(self, path_to_mnist: str) -> None:
        """Load dataset from file.

        Args:
            path_to_mnist: path to folder containing mnist.
        """
        with open(os.path.join(path_to_mnist, "mnist_train.npy"), "rb") as f:
            self.x_train = cp.load(f)
            self.y_train = cp.load(f)

        with open(os.path.join(path_to_mnist, "mnist_test.npy"), "rb") as f:
            self.x_test = cp.load(f)
            self.y_test = cp.load(f)

    def train_iter(self) -> None:
        """Train model for one epoch."""
        dataloader = self.datafeeder(self.x_train, self.y_train, True)
        train_error = []
        total_len = len(self.y_train) // self.batch_size
        for batch in tqdm.tqdm(dataloader, total=total_len):
            x, y = batch
            x = x.transpose(1, 0)
            x = x.reshape(self.batch_size, 1, 28, 28)
            y = y[:, cp.newaxis]
            y_hat = self.model.forward(x)
            loss = self.loss_function.forward(y_hat, y)
            error = self.loss_function.backward(y)
            self.model.backward(error)
            self.model.update_params()
            train_error.append(loss)
        print(cp.mean(train_error))

    def test_iter(self) -> None:
        """Test model."""
        test_dataloader = self.datafeeder(self.x_test, self.y_test)
        test_error = []
        epoch_tp = 0
        epoch_total = 0
        total_len = len(self.y_test) // self.batch_size
        for batch in tqdm.tqdm(test_dataloader, total=total_len):
            x, y = batch
            x = x.transpose(1, 0)
            x = x.reshape(self.batch_size, 1, 28, 28)
            y_hat = self.model.forward(x)
            loss = self.loss_function.forward(y_hat, y)
            y_pred = cp.argmax(y_hat, axis=-1)
            correct = cp.sum(y_pred == y)
            total = cp.size(y)
            epoch_tp += correct
            epoch_total += total
            test_error.append(loss)
        print("test error", cp.mean(test_error))
        print("test acc", epoch_tp / epoch_total)

    def train_model(self) -> None:
        """Train model."""
        self.model = ViT(
            im_dim=(1, 28, 28),
            n_patches=7,
            h_dim=8,
            n_heads=2,
            num_blocks=2,
            classes=10,
        )
        self.loss_function = CrossEntropyLoss()
        self.optimizer = Adam()  # SGD()
        self.model.init_optimizer(self.optimizer)
        for epoch in range(self.epochs):
            self.train_iter()
            if epoch % self.test_epoch_interval == 0:
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
    parser.add_argument("--epochs", dest="epochs", required=False, default=6)
    parser.add_argument(
        "--test_epoch_interval", dest="test_epoch_interval", required=False, default=2
    )
    args = parser.parse_args()
    return (args.path_to_mnist, args.batch_size, args.epochs, args.test_epoch_interval)


if __name__ == "__main__":
    path_to_mnist, batch_size, epochs, test_epoch_interval = parse_args()
    vit_mnist = ViTNumPy(path_to_mnist, batch_size, epochs, test_epoch_interval)
    vit_mnist.train_model()
