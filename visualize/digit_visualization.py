import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

mnist_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
dataloader = torch.utils.data.DataLoader(
    mnist_data, batch_size=len(mnist_data), shuffle=True
)

# Get all images and labels
images, labels = next(iter(dataloader))

# Create a dictionary to store first occurrence of each digit
digit_examples = {}
for img, label in zip(images, labels):
    label = label.item()
    if label not in digit_examples and len(digit_examples) < 10:
        digit_examples[label] = img

# Sort by digit
digit_examples = dict(sorted(digit_examples.items()))

# Create subplot
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

# Plot each digit
for idx, (digit, img) in enumerate(digit_examples.items()):
    # Convert tensor to numpy and reshape
    img = img.numpy().reshape(28, 28)

    # Plot
    axes[idx].imshow(img, cmap="gray")
    axes[idx].axis("off")
    axes[idx].set_title(f"Digit: {digit}")

plt.tight_layout()
# Save the figure with high DPI for better quality
plt.savefig("mnist_digits.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the figure to free memory
