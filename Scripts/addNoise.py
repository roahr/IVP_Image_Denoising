import os
import numpy as np
import imageio.v2 as imageio
import torch
from tqdm import tqdm

def add_noise(image, sigma=50):
    image = torch.tensor(image / 255, dtype=torch.float32, device='mps')
    noise = torch.normal(0, sigma / 255, image.shape, device='mps')
    gauss_noise = image + noise
    return (gauss_noise * 255).cpu().numpy()

def save_image(image, path):
    image = np.round(np.clip(image, 0, 255)).astype(np.uint8)
    imageio.imwrite(path, image)

def crop_image(image, s=8):
    h, w, c = image.shape
    return image[:h - h % s, :w - w % s, :]

def process_images(input_dir, output_dir, sigma=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_list = os.listdir(input_dir)
    
    for filename in tqdm(file_list, desc="Processing Images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            img = imageio.imread(input_path)
            img = crop_image(img)
            img_noise = add_noise(img, sigma)
            save_image(img_noise, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_directory = "/Users/jayashre/Developer/VSC/IVP_Image_Denoising/FINAL/raw/DIV2K_valid_HR"
    output_directory = "/Users/jayashre/Developer/VSC/IVP_Image_Denoising/FINAL/Valid"
    process_images(input_directory, output_directory, sigma=50)
