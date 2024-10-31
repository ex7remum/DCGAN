import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, fm_size=64, out_channels=3):
        super(Generator, self).__init__()
        self.latent = latent_dim
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fm_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fm_size * 8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(fm_size * 8, fm_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_size * 4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(fm_size * 4, fm_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_size * 2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(fm_size * 2, fm_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_size),
            nn.ReLU(),
            
            nn.ConvTranspose2d(fm_size, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.shape[0], self.latent, 1, 1)
        return self.conv(z)
    
    
class Discriminator(nn.Module):
    def __init__(self, fm_size=64, in_channels=3):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, fm_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(fm_size, fm_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_size * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(fm_size * 2, fm_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_size * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(fm_size * 4, fm_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_size * 8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(fm_size * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)
