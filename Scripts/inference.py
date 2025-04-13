import os
import glob
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import imageio.v2 as imageio
import numpy as np

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
        return x - noise

def load_model(model_path, device):
    model = DnCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_image(model, image_path, device):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Process image
    with torch.no_grad():
        denoised = model(img_tensor)
    
    # Convert back to numpy array
    denoised = denoised.squeeze(0).cpu().numpy()
    denoised = np.transpose(denoised, (1, 2, 0))
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    
    return denoised

def process_directory(model, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files
    image_files = glob.glob(os.path.join(input_dir, '*.png'))
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Process image
            denoised = process_image(model, img_path)
            
            # Save result
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            imageio.imwrite(output_path, denoised)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    model_path = "/Users/jayashre/Developer/VSC/IVP_Image_Denoising/models/dncnn.pth"  # Update this to your model path
    input_dir = "/media/viplab/DATADRIVE1/NTIRE2025_ImageDenoising/FINAL/Validate"
    output_dir = "/media/viplab/DATADRIVE1/NTIRE2025_ImageDenoising/Results"
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Process images
    print("Processing images...")
    process_directory(model, input_dir, output_dir)
    
    print("Inference complete!") 