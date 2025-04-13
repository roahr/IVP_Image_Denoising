import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DIV2KDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, size=(64, 64), transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.clean_images = sorted(os.listdir(clean_dir))
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.size = size
        self.transform = transform if transform else T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])

        clean_img = Image.open(clean_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')

        clean_tensor = self.transform(clean_img)
        noisy_tensor = self.transform(noisy_img)

        return noisy_tensor, clean_tensor
