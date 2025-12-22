import json
import math
import os
from glob import glob

import albumentations as A
import pandas as pd
import numpy as np
import torch
from skimage import io
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms


csi_valid_subcarrier_index = []
csi_valid_subcarrier_index += [i for i in range(6, 32)]
csi_valid_subcarrier_index += [i for i in range(33, 59)]
CSI_SUBCARRIERS = len(csi_valid_subcarrier_index)


def encode_time(x, frequence_len, window_size):
    window_size *= 3
    frequencies = np.array([2**i for i in range(frequence_len)])
    x = x / window_size
    pos_enc = np.concatenate([
        np.sin(frequencies[:, None] * np.pi * x),
        np.cos(frequencies[:, None] * np.pi * x)
    ], axis=0)
    return pos_enc


class WificamDataset(Dataset):
    def __init__(self, data_dir, window_size, frequence_len):
        self.data_dir = data_dir
        self.window_size = window_size
        self.window_size_h = math.ceil(window_size / 2)
        self.frequence_len = frequence_len

        self.compute_statistics()

        data = pd.read_csv(os.path.join(data_dir, 'csi.csv'))
        csi = data['data']
        self.csi = np.zeros([len(csi), 256], dtype=np.float32)

        csi_complex = np.zeros([len(csi), CSI_SUBCARRIERS], dtype=np.complex64)
        for i in range(len(csi)):
            sample = np.array(json.loads(csi[i]), dtype=np.int64)
            for j in range(len(sample)):
                self.csi[i][j] = sample[j]
            for j in range(CSI_SUBCARRIERS):
                csi_complex[i][j] = complex(
                    sample[csi_valid_subcarrier_index[j]*2],
                    sample[csi_valid_subcarrier_index[j]*2-1]
                )

        self.ids = data['id']

        self.csi_amplitudes = np.abs(csi_complex)

        self.data_size = len(self.csi_amplitudes) - self.window_size

    def compute_statistics(self):
        total_pixels = 0
        sum_channels = np.zeros(3, dtype=np.float64)
        sum_squares_channels = np.zeros(3, dtype=np.float64)

        img_files = glob(os.path.join(self.data_dir, '*.png'))
        for file in tqdm(img_files, desc="Computing Stats"):
            img = io.imread(file).astype(np.float64)
            
            h, w, _ = img.shape
            num_pixels = h * w
            total_pixels += num_pixels

            sum_channels += np.sum(img, axis=(0, 1))
            sum_squares_channels += np.sum(np.square(img), axis=(0, 1))

        mean_per_channel = sum_channels  / total_pixels
        variance_per_channel = (sum_squares_channels / total_pixels) - np.square(mean_per_channel)
        std_per_channel = np.sqrt(variance_per_channel)

        self.normalized_mean = (mean_per_channel / 255.0).astype(np.float32)
        self.normalized_std = (std_per_channel / 255.0).astype(np.float32)

        mean_path = os.path.join(self.data_dir, 'mean.npy')
        std_path = os.path.join(self.data_dir, 'std.npy')
        np.save(mean_path, self.normalized_mean)
        np.save(std_path, self.normalized_std)

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        index = index + self.window_size_h
        spectrogram = self.csi_amplitudes[index-self.window_size_h:index+self.window_size_h-1]
        spectrogram = np.transpose(spectrogram, (1, 0))

        spectrogram_transforms = transforms.Compose([transforms.ToTensor()])
        spectrogram = spectrogram_transforms(spectrogram)

        image_ids = self.ids[index-self.window_size_h:index+self.window_size_h-1]

        image_ids = [
            id for id in image_ids
            if os.path.isfile(os.path.join(self.data_dir, str(id) + '.png'))
        ]

        id = image_ids[len(image_ids)//2]
        image_index = image_ids.index(id)

        image = io.imread(os.path.join(self.data_dir, str(id) + '.png'))
        image = A.Resize(128, 128, always_apply=True)(image=image)['image']
        image = torch.tensor(image, dtype=torch.float) / 255.0
        image = image.permute(2, 0, 1)
        image = transforms.Normalize(self.normalized_mean, self.normalized_std)(image)

        spectrogram_tenc = np.array([index])
        spectrogram_tenc = torch.tensor(encode_time(spectrogram_tenc, self.frequence_len, self.window_size)).unsqueeze(0).float()
        image_tenc = np.array([image_index])
        image_tenc = torch.tensor(encode_time(image_tenc, self.frequence_len, self.window_size)).unsqueeze(0).float()
        tenc = torch.cat((spectrogram_tenc, image_tenc), dim=2)
        return (tenc, spectrogram), (tenc, image)