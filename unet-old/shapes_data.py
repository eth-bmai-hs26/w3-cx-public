
import numpy as np
from skimage.draw import rectangle, disk
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

IMAGE_SIZE = 128

def generate_sample(image_size=IMAGE_SIZE):
    image = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros((image_size, image_size), dtype=np.float32)

    # Circle
    r = np.random.randint(10, 25)
    cy, cx = np.random.randint(r, image_size - r, size=2)
    rr, cc = disk((cy, cx), r)
    image[rr, cc] = 0.5

    # Quadratic rectangle (square)
    s = np.random.randint(20, 40)
    ry, rx = np.random.randint(0, image_size - s, size=2)
    rr, cc = rectangle(start=(ry, rx), extent=(s, s))
    image[rr, cc] = 1.0
    mask[rr, cc] = 1.0

    return image, mask

def print_training_data_set(dataset):
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    for i in range(10):
        img, mask = dataset[i]

        axes[0, i].imshow(img[0], cmap="gray")
        axes[1, i].imshow(mask[0], cmap="gray")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("Input")
    axes[1, 0].set_ylabel("Mask")

    plt.suptitle(f"Training dataset ({len(dataset)} images)")
    plt.show()

def display_results(test_img, pred, test_mask):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(test_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(pred > 0.5, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Ground truth")
    plt.imshow(test_mask, cmap="gray")
    plt.axis("off")

    plt.show()


class ShapesDataset(Dataset):
    def __init__(self, n_samples):
        # Generate the samples once, so every epoch sees the same images
        self.samples = [generate_sample() for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        return (
            torch.tensor(img).unsqueeze(0),
            torch.tensor(mask).unsqueeze(0)
        )
