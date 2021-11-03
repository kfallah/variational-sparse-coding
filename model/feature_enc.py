import torch
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