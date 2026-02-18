# prompt: To create each batch, the images should be randomly cropped (but consistent for rgb, depth and encoded) so that each crop has a size of 256x256

import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

class ConsistentRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        # Get image size
        w, h = img.size

        # Generate random crop coordinates
        left = random.randint(0, w - self.size[0])
        top = random.randint(0, h - self.size[1])
        right = left + self.size[0]
        bottom = top + self.size[1]

        # Crop the image
        img = img.crop((left, top, right, bottom))

        return img, (left, top, right, bottom) # Return image and crop parameters

def depth_to_tensor(depth_img):
    arr = np.array(depth_img, dtype=np.float32)  # keep precision
    arr /= 65535.0  # normalize to 0-1
    # arr = arr*65535
    return torch.from_numpy(arr).unsqueeze(0)  # add channel dimension

def rgb_to_tensor(rgb_img):
    arr = np.array(rgb_img, dtype=np.float32)  # keep precision
    arr /= 255.0  # normalize to 0-1
    # arr = arr*65535
    return torch.from_numpy(arr).permute(2,0,1) # .unsqueeze(0)


def depth_to_tensor_norm(depth_img):
    arr = np.array(depth_img, dtype=np.float32)  # keep precision
    arr = np.round(arr/10)
    arr /= 6550.0  # normalize to 0-1
    # arr = arr*65535
    arr = np.clip(arr, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0) # Depth in [dm]

# Custom Dataset with consistent cropping
class ImageDataset(Dataset):
    def __init__(self, encoded_image_paths, rgb_image_paths, depth_image_paths, transform=None, crop_size=256):
        self.encoded_image_paths = encoded_image_paths
        self.rgb_image_paths = rgb_image_paths
        self.depth_image_paths = depth_image_paths
        self.transform = transform
        self.crop_size = crop_size
        self.consistent_crop = ConsistentRandomCrop(crop_size)

    def __len__(self):
        return len(self.encoded_image_paths)

    def __getitem__(self, idx):
        encoded_image = Image.open(self.encoded_image_paths[idx]).convert('RGB')
        rgb_image = Image.open(self.rgb_image_paths[idx]).convert('RGB')
        depth_image = Image.open(self.depth_image_paths[idx])#.convert('I')

        # Apply consistent random crop
        encoded_image_cropped, crop_params = self.consistent_crop(encoded_image)
        rgb_image_cropped = rgb_image.crop(crop_params)
        depth_image_cropped = depth_image.crop(crop_params)

        if self.transform:
            encoded_image_cropped = self.transform(encoded_image_cropped)
            rgb_image_cropped = self.transform(rgb_image_cropped)
            # For depth, we might need a different transform or apply the same.
            # Assuming the same ToTensor() and potentially normalization if needed for depth.

        depth_image_cropped = depth_to_tensor(depth_image_cropped) # self.transform(depth_image_cropped)


        return encoded_image_cropped, rgb_image_cropped, depth_image_cropped

class RGBDDatasetLoader(Dataset):
    def __init__(self, encoded_image_paths, rgb_image_paths, depth_image_paths, transform=None):
        self.encoded_image_paths = self._filter_images(encoded_image_paths)
        self.rgb_image_paths = self._filter_images(rgb_image_paths)
        self.depth_image_paths = self._filter_images(depth_image_paths)
        self.transform = transform

    def _filter_images(self, list_paths):
        valid_images = []
        for path in list_paths:
            try:
                with Image.open(path) as img:
                    if img.size == (256, 256):  # (width, height)
                        valid_images.append(path)
            except Exception as e:
                # Skip corrupted files
                print(f"Skipping {path}: {e}")
        return valid_images

    def __len__(self):
        return len(self.encoded_image_paths)

    def __getitem__(self, idx):
        encoded_image = Image.open(self.encoded_image_paths[idx]).convert('RGB')
        rgb_image = Image.open(self.rgb_image_paths[idx]).convert('RGB')
        depth_image = Image.open(self.depth_image_paths[idx])#.convert('I')

        if self.transform:
            encoded_image = self.transform(encoded_image)
            rgb_image = self.transform(rgb_image)
            # For depth, we might need a different transform or apply the same.
            # Assuming the same ToTensor() and potentially normalization if needed for depth.

        depth_image = depth_to_tensor_norm(depth_image) # self.transform(depth_image_cropped)
        rgb_image = rgb_to_tensor(rgb_image)
        encoded_image = rgb_to_tensor(encoded_image)
        return encoded_image, rgb_image, depth_image

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transformations if needed, e.g., Normalize
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Example normalization for RGB
])

# List all image files in each directory
def list_image_files(directory):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    return sorted(image_files) # Sort for consistency

# Write some information about the dataset and dataloader
def summarize_data(train_loader, train_dataset):
    # Number of batches
    num_batches = len(train_loader)
    print(f"Number of batches: {num_batches}")

    # Total number of samples in the dataset
    num_samples = len(train_loader.dataset)
    print(f"Total samples: {num_samples}")

    # Number of samples per batch
    batch_size = train_loader.batch_size
    print(f"Batch size: {batch_size}")

    # Get a single sample
    encoded_img, rgb_img, depth_img = train_dataset[0]

    print("Encoded image shape:", encoded_img.shape)
    print("RGB image shape:", rgb_img.shape)
    print("Depth image shape:", depth_img.shape)

def visualize_data(train_loader):
    # Get a batch from the training dataloader
    encoded_images, rgb_targets, depth_targets = next(iter(train_loader))

    # Number of images to display (up to batch size)
    num_images_to_display = min(4, encoded_images.shape[0])

    fig, axes = plt.subplots(num_images_to_display, 3, figsize=(10, num_images_to_display * 4))

    for i in range(num_images_to_display):
        # Display RGB image
        ax0 = axes[i, 0] if num_images_to_display > 1 else axes[0]
        ax0.imshow(rgb_targets[i].permute(1, 2, 0))  # Permute dimensions for matplotlib
        ax0.set_title('RGB Target')
        ax0.axis('off')

        # Display Depth image
        ax1 = axes[i, 1] if num_images_to_display > 1 else axes[1]
        # For grayscale depth, you might need to squeeze the channel dimension
        ax1.imshow(depth_targets[i].squeeze().cpu().numpy(), cmap='gray')
        print(depth_targets[i].squeeze().cpu().numpy().min(), depth_targets[i].squeeze().cpu().numpy().max())
        print(rgb_targets[i].squeeze().cpu().numpy().min(), rgb_targets[i].squeeze().cpu().numpy().max())
        print(encoded_images[i].squeeze().cpu().numpy().min(), encoded_images[i].squeeze().cpu().numpy().max())
        ax1.set_title('Depth Target')
        ax1.axis('off')

        # Display Encoded image
        ax2 = axes[i, 2] if num_images_to_display > 1 else axes[2]
        ax2.imshow(encoded_images[i].permute(1, 2, 0))  # Permute dimensions for matplotlib
        ax2.set_title('Encoded Image')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()