import pytorch_lightning as pl
import torch
from torch import nn

from ptp.models.utils import num_trainable_params


def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size, c, d, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1, 1)).repeat(1, c, d, h, w).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(batch_size, -1)
    # gradient_norm = gradient.norm(2, dim=1)
    # Compute norm manually
    gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-12)

    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


# TODO: spectral normalization?
class Discriminator(pl.LightningModule):

    def __init__(self, input_channels):
        super().__init__()
        # input: 1, 256, 256, 256
        self.layers = nn.ModuleList([
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=4, padding=1, bias=False),  # 16, 128, 128, 128
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 32, 64, 64, 64
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 64, 32, 32, 32
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 128, 16, 16, 16
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 128, 8, 8, 8
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 256, 4, 4, 4
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(2048, 1)
        ])


    def forward(self, x):
        batch_size = x.shape[0]
        for l in self.layers:
            x = l(x)
        x = x.view(batch_size, -1)
        for l in self.linears:
            x = l(x)
        return x


if __name__ == '__main__':
    discriminator = Discriminator(1)
    input_pt = torch.rand((10, 1, 256, 256, 256))
    print(discriminator(input_pt).shape)
    print(num_trainable_params(discriminator))
