import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def sample_laplacian(b, device='cpu'):
    u = torch.rand(b.shape, device=device)-0.5
    z = -b * torch.sign(u) * torch.log((1.0-2.0*torch.abs(u)).clamp(min=1e-10))   
    return z

def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - torch.abs(zeta)) * torch.sign(c)

class MLPEncoder(nn.Module):

    def __init__(self, img_size, dict_size):
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
        self.scale = nn.Linear((img_size**2), dict_size)
        #self.shift = nn.Linear((img_size**2), dict_size)

    def forward(self, x):
        feat = self.enc(x)
        return self.scale(feat), 0.#self.shift(feat)

class MLPEncoderLISTA(nn.Module):

    def __init__(self, img_size, dict_size, zeta):
        super(MLPEncoderLISTA, self).__init__()
        self.enc = nn.Sequential(
                        nn.Linear(img_size**2, (img_size**2) * 2),
                        nn.ReLU(),
                        nn.Linear((img_size**2) * 2, (img_size**2) * 4),
                        nn.ReLU(),
                        nn.Linear((img_size**2) * 4, (img_size**2) * 2),
                        nn.ReLU(),
                        nn.Linear((img_size**2) * 2, (img_size**2)),
                        nn.ReLU(),
                        nn.Linear((img_size**2), dict_size))
        self.lambda_ = torch.tensor(zeta)
        #self.lambda_ = nn.Parameter(torch.tensor(zeta), requires_grad=True)
        #self.W1 = nn.Linear(latent_dim, dict_size)
        self.W = nn.Sequential(
                        nn.Linear(dict_size, dict_size * 2),
                        nn.ReLU(),
                        nn.Linear(dict_size * 2, dict_size),
                        nn.ReLU(),
                        nn.Linear(dict_size, dict_size))

    def forward(self, x, A, iters=5):
        b_logspread = self.enc(x)
        b_spread = (0.5*b_logspread).exp()
        b = sample_laplacian(b_spread, x.device)
        b_refine = soft_threshold(b, self.lambda_)

        for i in range(iters):
            b_refine = soft_threshold(b + self.W(b_refine), self.lambda_)
            
        return b_spread, b_refine
