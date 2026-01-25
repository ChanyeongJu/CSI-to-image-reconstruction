import os
from pathlib import Path

import pytorch_lightning as L
import torch
from torch.utils.data import Subset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

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
data_dir = os.path.join(project_root, 'data', 'data_2026_01_20_mesh')

window_size = 151
batch_size = 32
epochs = 200
persistent_workers = True if num_workers > 0 else False


def train():
    dataset = WificamDataset(data_dir, window_size)

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


if __name__ == '__main__':
    train()
