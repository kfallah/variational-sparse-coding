import numpy as np
import torch.nn as nn

class MLPEncoder(nn.Module):

    def __init__(self, img_size):
        super(MLPEncoder, self).__init__()
        self.enc = nn.Sequential(
                nn.Linear(img_size**2, (img_size**2) * 2),
                nn.ReLU(),
                nn.Linear((img_size**2) * 2, (img_size**2) * 4),
                nn.ReLU(),
                nn.Linear((img_size**2) * 4, (img_size**2) * 2),
                nn.ReLU(),
                nn.Linear((img_size**2) * 2, (img_size**2)),
                nn.ReLU())    

    def forward(self, x):
        return self.enc(x)

class ConvEncoder(nn.Module):

    def __init__(self, num_feat, num_channels):
        super(ConvEncoder, self).__init__()
        self.enc = nn.Sequential(
                # 32 x 32
                nn.Conv2d(num_channels, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # 16 x 16
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 8 x 8
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 1 x 1
                nn.Conv2d(128, 256, 5, 4, 0),
                nn.BatchNorm2d(256),
                nn.ReLU()               
        )
        self.linear = nn.Linear(256, num_feat)


    def forward(self, x):
        z = self.enc(x).reshape(len(x), -1)
        return self.linear(z)

class ConvDecoder(nn.Module):
    def __init__(self, num_feat, num_channels):
        super(ConvDecoder, self).__init__()
        self.linear = nn.Linear(num_feat, 256*4*4)
        self.enc = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, num_channels, 4, 2, 1),
        )

    def forward(self, x):
        # If decoding multiple samples per image, handle that across batches
        if x.dim() > 2:
            x_hat = self.linear(x).reshape((len(x) * x.shape[1]), -1, 4, 4)
            x_hat = self.enc(x_hat)
            return x_hat.reshape(len(x), x.shape[1], *x_hat.shape[1:])
        else:
            x_hat = self.linear(x).reshape(len(x), -1, 4, 4)
            return self.enc(x_hat)
