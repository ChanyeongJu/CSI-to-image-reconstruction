import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as L
import cv2
from torch.utils.data import Subset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision import transforms

from dataset import WificamDataset
from mopoevae import MoPoEVAE


model_dir = 'runs/mopoevae_ct'
model_name = 'bestLoss'
data_dir = 'data/20251017'

if torch.backends.mps.is_available():
    device = torch.device('mps')
    accelerator = 'mps'
elif torch.cuda.is_available():
    device = torch.device('cuda')
    accelerator = 'gpu'
else:
    device = torch.device('cpu')
    accelerator = 'cpu'

num_workers = 0
batch_size = 32
window_size = 151
frequence_len = 8
learning_rate = 1e-3
z_dim = 128
aggregate_method = 'concat'


def train():
    dataset = WificamDataset(data_dir, window_size, frequence_len)

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=False)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5, shuffle=False)

    dataset_train = Subset(dataset, train_idx)
    dataset_val = Subset(dataset, val_idx)
    dataset_test = Subset(dataset, test_idx)

    dataloader_train = DataLoader(ConcatDataset([dataset_train]), batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dataloader_val = DataLoader(ConcatDataset([dataset_val]), batch_size=batch_size*8, shuffle=False, num_workers=num_workers, drop_last=True)
    dataloader_test = DataLoader(ConcatDataset([dataset_test]), batch_size=batch_size*8, shuffle=False, num_workers=num_workers, drop_last=True)

    model = MoPoEVAE(
        weight_ll=True,
        lr=learning_rate,
        sequence_length=window_size,
        z_dim=z_dim,
        frequence_L=frequence_len,
        aggregate_method=aggregate_method,
        imgMean=dataset.normalized_mean,
        imgStd=dataset.normalized_std,
        log=False,
    )

    callbacks = [
        ModelCheckpoint(monitor='val_loss', mode='min', save_last=False, filename=model_name, dirpath=model_dir),
        EarlyStopping(monitor="val_loss", mode='min', patience=25, min_delta=10.0, verbose=True),
    ]

    trainer = L.Trainer(accelerator=accelerator, gradient_clip_val=1.0, logger=None, callbacks=callbacks, max_epochs=50)
    trainer.fit(model, dataloader_train, dataloader_val)

    img_mean = np.load(os.path.join(data_dir, 'mean.npy'))
    img_std = np.load(os.path.join(data_dir, 'std.npy'))
    model = MoPoEVAE.load_from_checkpoint(
        f'{model_dir}/{model_name}.ckpt',
        weight_ll=True,
        lr=learning_rate,
        sequence_length=window_size,
        z_dim=z_dim,
        frequence_L=frequence_len,
        aggregate_method=aggregate_method,
        map_location=device,
        imgMean=img_mean,
        imgStd=img_std,
        log=None,
    )
    model.to(device)
    model.eval()

    img_mean = model.imgMean.reshape(1, 3, 1, 1).cpu().numpy()
    img_std = model.imgStd.reshape(1, 3, 1, 1).cpu().numpy()


def test():
    csi_valid_subcarrier_index = []
    csi_valid_subcarrier_index += [i for i in range(6, 32)]
    csi_valid_subcarrier_index += [i for i in range(33, 59)]
    CSI_SUBCARRIERS = len(csi_valid_subcarrier_index)

    img_mean = np.load(os.path.join(data_dir, 'mean.npy'))
    img_std = np.load(os.path.join(data_dir, 'std.npy'))

    model = MoPoEVAE.load_from_checkpoint(
        f'{model_dir}/{model_name}.ckpt',
        weight_ll=True,
        lr=learning_rate,
        sequence_length=window_size,
        z_dim=z_dim,
        frequence_L=frequence_len,
        aggregate_method=aggregate_method,
        map_location=device,
        imgMean=img_mean,
        imgStd=img_std,
        log=None,
    )
    model.to(device)
    model.eval()

    model = torch.compile(model)

    img_mean = img_mean.reshape(1, 3, 1, 1)
    img_std = img_std.reshape(1, 3, 1, 1)

    data = pd.read_csv(os.path.join(data_dir, 'csi.csv'))
    csi = data['data']

    csi_complex = np.zeros([len(csi), CSI_SUBCARRIERS], dtype=np.complex64)
    for i in range(len(csi)):
        sample = np.array(json.loads(csi[i]), dtype=np.int64)
        for j in range(CSI_SUBCARRIERS):
            csi_complex[i][j] = complex(
                sample[csi_valid_subcarrier_index[j]*2],
                sample[csi_valid_subcarrier_index[j]*2-1]
            )

    csi_amplitudes = np.abs(csi_complex)

    def encode_time(x, frequence_len, window_size):
        window_size *= 3
        frequencies = np.array([2**i for i in range(frequence_len)])
        x = x / window_size
        pos_enc = np.concatenate([
            np.sin(frequencies[:, None] * np.pi * x),
            np.cos(frequencies[:, None] * np.pi * x)
        ], axis=0)
        return pos_enc
     
    for i in range(len(csi_amplitudes) - window_size):
        spectrogram = csi_amplitudes[i:i+window_size]
        spectrogram = np.transpose(spectrogram, (1, 0))

        spectrogram_transforms = transforms.Compose([transforms.ToTensor()])
        spectrogram = spectrogram_transforms(spectrogram)
        spectrogram = torch.unsqueeze(spectrogram, dim=0)

        spectrogram_tenc = np.array([i])
        spectrogram_tenc = torch.tensor(encode_time(spectrogram_tenc, frequence_len, window_size)).unsqueeze(0).float()
        image_tenc = np.array([30])
        image_tenc = torch.tensor(encode_time(image_tenc, frequence_len, window_size)).unsqueeze(0).float()
        tenc = torch.cat((spectrogram_tenc, image_tenc), dim=2)
        tenc = torch.unsqueeze(tenc, dim=0)

        spectrogram = tenc.to(device), spectrogram.to(device)
        with torch.no_grad():
            reconstruction = model.decode(model.encode_subset([spectrogram], [0]))[1][0][1]

        reconstruction = reconstruction.permute(0, 2, 3, 1).cpu().numpy()

        pred_sample = reconstruction[0][..., ::-1]
        pred_sample = pred_sample.transpose(2, 0, 1)
        pred_sample = pred_sample * img_std + img_mean
        pred_sample = np.clip(pred_sample, 0, 1)
        pred_sample = (pred_sample * 255).astype(np.uint8)
        pred_sample = pred_sample[0]
        pred_sample = pred_sample.transpose(1, 2, 0)

        plt.imshow(pred_sample)
        plt.show()


if __name__ == '__main__':
    train()
    test()