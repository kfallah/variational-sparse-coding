import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F

import model.lista as lista
from model.feature_enc import MLPEncoder

def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    kl_loss = -0.5 * torch.mean(1 + logvar - (mu ** 2) - logvar.exp())
    return kl_loss, mu + eps*std

def sample_laplacian(mu, logscale):
    u = torch.rand(logscale.shape, device=logscale.device) -0.5
    scale = (logscale).exp()
    z = mu - scale * torch.sign(u) * torch.log((1.0-2.0*torch.abs(u)).clamp(min=1e-10))   
    kl_loss = (-logscale + mu.abs() + scale*(-mu.abs() / scale).exp()).mean()
    return kl_loss, z

def sample_spikeslab(mu, logscale, logspike, c=50, alpha=0.2):
    # From first submission of (Tonolini et al 2020) without pseudo-inputs
    # Code found https://github.com/Alfo5123/Variational-Sparse-Coding/blob/master/src/models/vsc.py
    std = torch.exp(0.5*logscale)
    eps = torch.randn_like(std)
    gaussian = eps.mul(std).add_(mu)
    eta = torch.rand_like(std)
    selection = F.sigmoid(c*(eta + logspike.exp() - 1))
    z = selection * gaussian

    spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 
    kl_loss = -0.5 * 1e-2 * torch.sum(spike.mul(1 + logscale - mu.pow(2) \
                                        - logscale.exp())) + \
                    1e-1 * (torch.sum((1 - spike).mul(torch.log((1 - spike) \
                                            /(1 - alpha))) + \
                    spike.mul(torch.log(spike / alpha))))

    return kl_loss / len(z), z

class VIEncoder(nn.Module):

    def __init__(self, img_size, dict_size, solver_args):
        super(VIEncoder, self).__init__()
        self.solver_args = solver_args
        if self.solver_args.feature_enc == "MLP":
            self.enc = MLPEncoder(img_size)
        else:
            raise NotImplementedError
        self.scale = nn.Linear((img_size**2), dict_size)
        self.shift = nn.Linear((img_size**2), dict_size)
        if self.solver_args.prior == "spikeslab":
            self.spike = nn.Linear((img_size**2), dict_size)
            self.c = 50
            self.alpha = 0.15

    def forward(self, x, A):
        feat = self.enc(x)
        b_logscale = self.scale(feat)
        b_shift = self.shift(feat)

        if self.solver_args.prior == "laplacian":
            kl_loss, b = sample_laplacian(b_shift, b_logscale)
            b = lista.soft_threshold(b, torch.tensor(0.1, device=b.device))
        elif self.solver_args.prior == "gaussian":
            kl_loss, b = sample_gaussian(b_shift, b_logscale)
        elif self.solver_args.prior == "spikeslab":
            logspike = -F.relu(-self.spike(feat))
            kl_loss, b = sample_spikeslab(b_shift, b_logscale, logspike, self.c, self.alpha)
            return logspike, kl_loss, b
        else:
            raise NotImplementedError
        return kl_loss, b