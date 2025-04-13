import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.multiprocessing as mp
from tqdm import tqdm

# ✅ 1. DnCNN Architecture
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # remove predicted noise from input

# ✅ 2. Custom Dataset
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_images = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))
        self.clean_images = sorted(glob.glob(os.path.join(clean_dir, '*.png')))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy = Image.open(self.noisy_images[idx]).convert('RGB')
        clean = Image.open(self.clean_images[idx]).convert('RGB')
        return self.transform(noisy), self.transform(clean)

# ✅ 3. Training Function
def train_model(noisy_path, clean_path, save_path='dncnn.pth', epochs=100, batch_size=8, lr=1e-3, patch_size=64):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cpu")

    dataset = DenoisingDataset(noisy_path, clean_path,
        transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    model = DnCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        loss_epoch = 0
        for noisy, clean in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss_epoch/len(dataloader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()

    noisy_images_dir = "/Users/jayashre/Developer/VSC/IVP_Image_Denoising/FINAL/Train"
    clean_images_dir = "/Users/jayashre/Developer/VSC/IVP_Image_Denoising/FINAL/raw/DIV2K_train_HR"

    train_model(noisy_images_dir, clean_images_dir)

    
# mp.freeze_support()
# noisy_images_dir = "/Users/jayashre/Developer/VSC/IVP_Image_Denoising/FINAL/raw/DIV2K_train_HR"
# clean_images_dir = "/Users/jayashre/Developer/VSC/IVP_Image_Denoising/FINAL/raw/DIV2K_train_HR"
# train_model(noisy_images_dir, clean_images_dir)