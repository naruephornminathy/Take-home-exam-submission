import os
import random

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, Subset


def augment(image, mask):
    """Random flips/rotations applied to all 4 bands + mask together;
    color jitter (brightness) applied to RGB channels only, leaving
    NIR untouched as required."""
    # Random horizontal flip
    if random.random() > 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])

    # Random vertical flip
    if random.random() > 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])

    # Random 90 rotation
    if random.random() > 0.5:
        k = random.choice([1, 2, 3])
        image = torch.rot90(image, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])

    # Color jitter (brightness) (RGB)
    if random.random() > 0.5:
        rgb = image[:3]
        nir = image[3:4]
        brightness_factor = random.uniform(0.8, 1.2)
        rgb = torch.clamp(rgb * brightness_factor, 0, 1)
        image = torch.cat([rgb, nir], dim=0)

    return image, mask


class CloudDataset(Dataset):
    def __init__(self, red_dir, green_dir, blue_dir, nir_dir, gt_dir=None,
                 file_list=None, augment_data=False):
        self.red_dir = red_dir
        self.green_dir = green_dir
        self.blue_dir = blue_dir
        self.nir_dir = nir_dir
        self.gt_dir = gt_dir
        self.filenames = file_list if file_list is not None else sorted(os.listdir(red_dir))
        self.augment_data = augment_data

    def __len__(self):
        return len(self.filenames)

    def _band_path(self, band_dir, band_name, red_filename):
        suffix = red_filename.split('_', 1)[1]
        return os.path.join(band_dir, f'{band_name}_{suffix}')

    def _read_band(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)

    def __getitem__(self, idx):
        red_filename = self.filenames[idx]
        red = self._read_band(os.path.join(self.red_dir, red_filename))
        green = self._read_band(self._band_path(self.green_dir, 'green', red_filename))
        blue = self._read_band(self._band_path(self.blue_dir, 'blue', red_filename))
        nir = self._read_band(self._band_path(self.nir_dir, 'nir', red_filename))

        image = np.stack([red, green, blue, nir], axis=0) / 65535.0
        image = torch.from_numpy(image).float()

        if self.gt_dir is not None:
            mask = self._read_band(self._band_path(self.gt_dir, 'gt', red_filename))
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0).float()

            if self.augment_data:
                image, mask = augment(image, mask)

            return image, mask

        return image, red_filename


def _load_filenames_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return [f'red_{name}.TIF' for name in df['name']]


def get_train_val_loaders(root_dir, batch_size=8, val_split=0.2, seed=42,
                           num_workers=2, csv_path=None, use_augmentation=False):
    red_dir = os.path.join(root_dir, 'train_red')
    green_dir = os.path.join(root_dir, 'train_green')
    blue_dir = os.path.join(root_dir, 'train_blue')
    nir_dir = os.path.join(root_dir, 'train_nir')
    gt_dir = os.path.join(root_dir, 'train_gt')

    if csv_path is None:
        csv_path = os.path.join(root_dir, 'training_patches_38-Cloud.csv')
    file_list = _load_filenames_from_csv(csv_path)

    dataset_no_aug = CloudDataset(red_dir, green_dir, blue_dir, nir_dir, gt_dir,
                                   file_list=file_list, augment_data=False)
    dataset_aug = CloudDataset(red_dir, green_dir, blue_dir, nir_dir, gt_dir,
                                file_list=file_list, augment_data=use_augmentation)

    val_size = int(len(dataset_no_aug) * val_split)
    train_size = len(dataset_no_aug) - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset_no_aug), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset_aug, train_indices)
    val_dataset = Subset(dataset_no_aug, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_test_loader(root_dir, batch_size=8, num_workers=2, csv_path=None):
    red_dir = os.path.join(root_dir, 'test_red')
    green_dir = os.path.join(root_dir, 'test_green')
    blue_dir = os.path.join(root_dir, 'test_blue')
    nir_dir = os.path.join(root_dir, 'test_nir')

    if csv_path is None:
        csv_path = os.path.join(root_dir, 'test_patches_38-Cloud.csv')
    file_list = _load_filenames_from_csv(csv_path)

    test_dataset = CloudDataset(red_dir, green_dir, blue_dir, nir_dir, gt_dir=None, file_list=file_list)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
