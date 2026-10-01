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

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 2 * latent_dim, 3, padding=1),  # Keep spatial dims
        )

    def forward(self, x):
        h = self.down(x)  # shape: (B, 2*latent_dim, 8, 8)
        mu, logvar = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=4, out_channels=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),  # 16 -> 32
            nn.Tanh(),  # Normalize output between -1 and 1
        )

    def forward(self, z):
        return self.up(z)

def vae_loss(x_recon, x, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss * 0.01  # weight KL to balance training

class StableStyleVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
def train_vae(in_channels, latent_dim, train_dataset, batch_size, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    vae = StableStyleVAE(in_channels=in_channels, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device.type == "cuda" else 0,  # worker processes break in notebooks on macOS/Windows
        pin_memory=True if device.type == "cuda" else False,
    )

    vae.train()

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        total_loss = 0.0

        for batch in dataloader:
            imgs = batch[0].to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = vae(imgs)
            loss = vae_loss(x_recon, imgs, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    torch.save(vae.state_dict(), "vae.pth")