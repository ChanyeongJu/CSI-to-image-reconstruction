import json
import os
from glob import glob

import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


CSI_VALID_SUBCARRIER_INDEX = [i for i in range(6, 32)] + [i for i in range(33, 59)]
NUM_SUBCARRIERS = len(CSI_VALID_SUBCARRIER_INDEX)


class WificamDataset(Dataset):
    def __init__(self, base_dir, window_size):
        self.base_dir = base_dir
        self.window_size = window_size

        self.csi_amplitudes = []
        self.image_paths = []
        self.load_data()

    def load_data(self):
        data_paths = glob(os.path.join(self.base_dir, '**', 'csi.csv'), recursive=True)
        for data_path in data_paths:
            data_dir = os.path.dirname(data_path)
            df = pd.read_csv(data_path).sort_values(by='id')

            raw_csi = df['data'].apply(json.loads).values
            raw_csi = np.array([np.array(x, dtype=np.int32) for x in raw_csi])
            real = raw_csi[:, [i * 2 for i in CSI_VALID_SUBCARRIER_INDEX]]
            imag = raw_csi[:, [i * 2 - 1 for i in CSI_VALID_SUBCARRIER_INDEX]]
            amplitude = np.sqrt(real**2 + imag**2).astype(np.float32)

            all_image_files = glob(os.path.join(data_dir, '*.png'))
            readable_image_ids = []
            for f in all_image_files:
                test_img = cv2.imread(f)
                if test_img is not None:
                    try:
                        img_id = int(os.path.basename(f).split('.')[0])
                        readable_image_ids.append(img_id)
                    except ValueError:
                        continue

            if not readable_image_ids:
                print(f'Warning: No valid images found in {data_dir}. Skipping this directory.')
                continue

            readable_image_ids = np.array(sorted(readable_image_ids))

            num_samplies = len(amplitude) - self.window_size
            for i in range(num_samplies):
                target_id = df.iloc[i]['id'] + (self.window_size // 2)
                best_img_id = readable_image_ids[np.abs(readable_image_ids - target_id).argmin()]

                self.csi_amplitudes.append(amplitude[i:i + self.window_size])
                self.image_paths.append(os.path.join(data_dir, f'{best_img_id}.png'))
        
        self.csi_amplitudes = np.array(self.csi_amplitudes)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        spectrogram = torch.from_numpy(self.csi_amplitudes[index])

        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return spectrogram, image
