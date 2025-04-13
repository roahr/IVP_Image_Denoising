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
# from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast, GradScaler
import numpy as np

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Directories
CLEAN_DIR = '/media/viplab/DATADRIVE1/NTIRE2025_ImageDenoising/DIV2K/Train/DIV2K_train_HR'
NOISY_DIR = '/media/viplab/DATADRIVE1/NTIRE2025_ImageDenoising/FINAL/Train'
SAVE_DIR = '/media/viplab/DATADRIVE1/NTIRE2025_ImageDenoising/model_checkpoints/apple_ai'
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 1
EPOCHS = 50
LR = 2e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset & Loader
dataset = DIV2KDataset(CLEAN_DIR, NOISY_DIR, size=(192, 192))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, optimizer, scheduler
model = IPTV2().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.L1Loss()
scaler = GradScaler()

# Evaluation Metrics
def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# def calc_ssim(img1, img2):
#     img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
#     img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
#     return ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    psnr_list = []
    # ssim_list = []

    for noisy, clean in loader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            denoised = model(noisy)
            loss = criterion(denoised, clean)

        # Scales the loss, calls backward, unscales gradients, and clips if needed
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Metrics (only 1st image of batch)
        with torch.no_grad():
            psnr = calc_psnr(denoised[0], clean[0])
            # ssim_val = calc_ssim(denoised[0], clean[0])
            psnr_list.append(psnr)
            # ssim_list.append(ssim_val)

    scheduler.step()

    avg_loss = total_loss / len(loader)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    # avg_ssim = sum(ssim_list) / len(ssim_list)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

    # Save checkpoint
    torch.save(model.state_dict(), f"{SAVE_DIR}/iptv2_epoch{epoch+1}.pth")

    # Optional: clear unused memory
    torch.cuda.empty_cache()
