import math
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_swiss_roll
from torchvision.utils import save_image
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from IPython.display import display
from tqdm.auto import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class SwissRoll2D(Dataset):
    """
    Create a 2D Swiss Roll by taking (x, z) from the 3D Swiss roll.
    Optionally normalize each axis to have mean 0 and std 1.
    """

    def __init__(self, num_samples=10000, noise=0.05, normalize=True):
        data3d, _ = make_swiss_roll(n_samples=num_samples, noise=noise)
        xz = data3d[:, [0, 2]].astype(np.float32)
        x = torch.from_numpy(xz)
        if normalize:
            mu = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            x = (x - mu) / std
        self.data = x

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return self.data[i]


def data_limits(x, pad=0.1):
    x_np = x.detach().cpu().numpy()
    xmin, ymin = x_np.min(axis=0)
    xmax, ymax = x_np.max(axis=0)
    xr = xmax - xmin
    yr = ymax - ymin
    return xmin - pad * xr, xmax + pad * xr, ymin - pad * yr, ymax + pad * yr


def plot_points(x, title="Points", s=5, limits=None, figsize=(5, 5)):
    x_np = x.detach().cpu().numpy()
    if limits is None:
        limits = data_limits(x, pad=0.1)
    xmin, xmax, ymin, ymax = limits
    plt.figure(figsize=figsize)
    plt.scatter(x_np[:, 0], x_np[:, 1], s=s, alpha=0.7)
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect("equal", "box")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(alpha=0.2)
    plt.show()


def plot_forward_grid(model, x0, steps, cols=3, title="Forward (simple)"):
    x0 = x0.to(model.device)
    B = min(2000, x0.size(0))
    idx = torch.randperm(x0.size(0), device=model.device)[:B]
    x0_sub = x0[idx]

    xmin, xmax, ymin, ymax = data_limits(x0_sub, pad=0.1)
    steps = [int(s) for s in steps if 0 <= int(s) < model.T]
    panels = [("t = 0", x0_sub.cpu().numpy())]
    for tval in steps:
        tt = torch.full((B,), tval, device=model.device, dtype=torch.long)
        xt = model.q_sample(x0_sub, tt)
        panels.append((f"t = {tval}", xt.cpu().numpy()))

    rows = math.ceil(len(panels) / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (label, pts) in enumerate(panels, start=1):
        plt.subplot(rows, cols, i)
        plt.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.7)
        plt.title(label)
        ax = plt.gca()
        ax.set_aspect("equal", "box")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.grid(alpha=0.2)
    plt.suptitle(title)
    plt.show()


def plot_reverse_grid(model, steps, ref_points, n_show=3000, cols=3, title="Reverse"):
    steps = [int(s) for s in steps if 0 <= int(s) < model.T]
    if 0 not in steps:
        steps.append(0)
    steps = sorted(set(steps), reverse=True)

    n = max(n_show, 2000)
    _, snaps = model.sample(n=n, step_size=1, eta=1.0, collect_ts=steps)

    xmin, xmax, ymin, ymax = data_limits(ref_points, pad=0.1)
    rows = math.ceil(len(steps) / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, tval in enumerate(steps, start=1):
        x_t = snaps[tval][:n_show].cpu().numpy()
        plt.subplot(rows, cols, i)
        plt.scatter(x_t[:, 0], x_t[:, 1], s=5, alpha=0.7)
        plt.title(f"t = {tval}")
        ax = plt.gca()
        ax.set_aspect("equal", "box")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.grid(alpha=0.2)
    plt.suptitle(title)
    plt.show()