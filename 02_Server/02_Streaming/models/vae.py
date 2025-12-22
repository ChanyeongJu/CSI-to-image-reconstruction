import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.distributions import Normal as Normal


z_dim = 128


class VAE(L.LightningModule):
    def __init__(self, window_size, num_subcarriers):
        super(VAE, self).__init__()
        self.encoder = Encoder(window_size, num_subcarriers)
        self.decoder = Decoder()

    def encode(self, x):
        mu, logvar = self.encoder.encode(x)
        scale = torch.exp(0.5 * logvar)
        qz_x = Normal(loc=mu, scale=scale)
        return qz_x
    
    def decode(self, qz_x):
        if self.training:
            z = qz_x.rsample()
        else:
            z = qz_x.loc

        x_hat = self.decoder.decode(z)
        return x_hat

    def forward(self, x):
        qz_x = self.encode(x)
        x_hat = self.decode(qz_x)
        return qz_x, x_hat

    def loss_function(self, x, qz_x, x_hat):
        kl = self.calc_kl(qz_x)
        mse = self.calc_mse(x, x_hat)
        total = kl + mse
        losses = {"loss": total, "kl": kl, 'mse': mse}
        return losses

    def calc_kl(self, qz_x):
        variance = qz_x.scale**2
        kl = 0.5 * (variance + qz_x.loc**2 - 1 - torch.log(variance))
        return kl.mean(0).sum()
    
    def calc_mse(self, x, x_hat):
        diff = x - x_hat
        squared_diff = diff.pow(2)
        mse = squared_diff.mean(0).sum()
        return mse

    def __step(self, batch, batch_idx, stage):
        qz_x, x_hat = self.forward(batch[0])
        loss = self.loss_function(batch[1], qz_x, x_hat)
        for loss_n, loss_val in loss.items():
            self.log(f"{stage}_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=False)
        return loss["loss"]

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.__step(batch, batch_idx, stage="val")
  
    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3, amsgrad=True)
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        modules = []
        hidden_dims = [512, 256, 192, 128, 96, 48]
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.01),
                )
            )
        else:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
                    nn.Sigmoid(),
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 4)

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 512, 2, 2)
        image = self.decoder(z)
        return image


class Encoder(nn.Module):
    def __init__(self, window_size, num_subcarriers):
        super(Encoder, self).__init__()

        self.subcarrier_encoder = nn.Sequential(
            nn.Linear(num_subcarriers, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 8),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(8 * window_size, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Sequential(
            nn.Linear(256, z_dim),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.subcarrier_encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.latent_encoder(x)

        mu = self.mu(x)
        logvar = self.logvar(x) * 10.0
        return mu, logvar
