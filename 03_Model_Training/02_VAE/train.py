import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as L
import numpy as np
import torch
import cv2
from torch.utils.data import Subset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import WificamDataset, NUM_SUBCARRIERS
from vae import VAE


num_workers = 2
torch.set_num_threads(4)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    accelerator = 'mps'
elif torch.cuda.is_available():
    device = torch.device('cuda')
    accelerator = 'gpu'
else:
    device = torch.device('cpu')
    accelerator = 'cpu'

current_file_path = Path(__file__).resolve()
current_folder = current_file_path.parent
project_root = current_folder.parent
data_dir = os.path.join(project_root, 'data', '2025_12_22')
test_dir = os.path.join(project_root, 'data', '2025_12_22_test')

window_size = 151
batch_size = 32
epochs = 100
persistent_workers = True if num_workers > 0 else False


def train():
    dataset = WificamDataset(data_dir, window_size)
    
    csi_amplitudes = dataset.csi_amplitudes[:, 0, :].T
    plt.imshow(csi_amplitudes[:, :2000], cmap='viridis', aspect='auto')
    plt.show()

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=False)

    dataset_train = Subset(dataset, train_idx)
    dataset_val = Subset(dataset, val_idx)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    output_dir = os.path.join(current_folder, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    model = VAE(window_size=window_size, num_subcarriers=NUM_SUBCARRIERS)

    callbacks = [
        ModelCheckpoint(
            monitor='val_loss', 
            mode='min', 
            save_top_k=-1,
            save_last=False, 
            filename='{epoch}-{val_loss:.4f}',
            dirpath=output_dir
        ),
    ]

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=1.0,
        logger=False,
        callbacks=callbacks,
        max_epochs=epochs
    )
    trainer.fit(model, dataloader_train, dataloader_val)


def test():
    dataset = WificamDataset(test_dir, window_size)
    dataloader_test = DataLoader(
        dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    output_dir = os.path.join(current_folder, 'outputs')
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, 'epoch=25-val_loss=384.1094.ckpt')

    model = VAE.load_from_checkpoint(
        checkpoint_path,
        window_size=window_size,
        num_subcarriers=NUM_SUBCARRIERS,
    )

    model.to(device)
    model.eval()
    
    idx = 0
    for batch in tqdm(dataloader_test):
        spectrogram, image = batch
        spectrogram = spectrogram.to(device)
        image = image.to(device)

        with torch.no_grad():
            reconstruction = model.decode(model.encode(spectrogram))

        image = image.permute(0, 2, 3, 1).cpu().numpy()
        reconstruction = reconstruction.permute(0, 2, 3, 1).cpu().numpy()

        for i in range(len(reconstruction)):
            data_sample = image[i][..., ::-1]
            data_sample = np.clip(data_sample, 0, 1)
            data_sample = (data_sample * 255).astype(np.uint8)

            pred_sample = reconstruction[i][..., ::-1]
            pred_sample = np.clip(pred_sample, 0, 1)
            pred_sample = (pred_sample * 255).astype(np.uint8)

            combined_image = np.concatenate((data_sample, pred_sample), axis=1)

            cv2.imwrite(os.path.join(image_dir, f'{str(idx)}.png'), combined_image)
            idx += 1


if __name__ == '__main__':
    train()
    test()
