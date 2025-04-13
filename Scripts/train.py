import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.iptv2 import IPTV2
from utils.dataloader import DIV2KDataset
import torchvision.transforms as T
import torchvision.utils as vutils
from math import log10
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Directories
CLEAN_DIR = '/Users/jayashre/Developer/VSC/IVP_Image_Denoising/FINAL/raw/DIV2K_train_HR'
NOISY_DIR = '/Users/jayashre/Developer/VSC/IVP_Image_Denoising/FINAL/Train'
SAVE_DIR = '/Users/jayashre/Developer/VSC/IVP_Image_Denoising/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 2
EPOCHS = 50
LR = 2e-4
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Dataset & Loader
dataset = DIV2KDataset(CLEAN_DIR, NOISY_DIR, size=(256, 256))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, optimizer, scheduler
model = IPTV2().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.L1Loss()

# Evaluation Metrics
def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_ssim(img1, img2):
    img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
    return ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    psnr_list = []
    ssim_list = []

    for noisy, clean in loader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        denoised = model(noisy)
        loss = criterion(denoised, clean)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Metrics (only 1st image of batch)
        with torch.no_grad():
            psnr = calc_psnr(denoised[0], clean[0])
            ssim_val = calc_ssim(denoised[0], clean[0])
            psnr_list.append(psnr)
            ssim_list.append(ssim_val)

    scheduler.step()

    avg_loss = total_loss / len(loader)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"{SAVE_DIR}/iptv2_epoch{epoch+1}.pth")
