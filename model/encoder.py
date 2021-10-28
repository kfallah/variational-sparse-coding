import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    kl_loss = -0.5 * torch.sum(1 + logvar - (mu ** 2) - logvar.exp())
    return kl_loss, mu + eps*std

def sample_laplacian(b):
    u = torch.rand(b.shape, device=b.device)-0.5
    z = -b * torch.sign(u) * torch.log((1.0-2.0*torch.abs(u)).clamp(min=1e-10))   
    kl_loss = 0.5 * (b - torch.log(b) - 1.0).mean()
    return kl_loss, z

def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - torch.abs(zeta)) * torch.sign(c)
    
class MLPEncoder(nn.Module):

    def __init__(self, img_size):
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

class VIEncoder(nn.Module):

    def __init__(self, img_size, dict_size, prior):
        super(VIEncoder, self).__init__()
        self.prior = prior
        self.enc = MLPEncoder(img_size)
        self.scale = nn.Linear((img_size**2), dict_size)
        if self.prior == "gaussian":
            self.shift = nn.Linear((img_size**2), dict_size)

    def forward(self, x):
        feat = self.enc(x)
        b_logscale = self.scale(feat)

        if self.prior == "laplacian":
            b_scale = (b_logscale * 0.5).exp()
            kl_loss, b = sample_laplacian(b_scale)
        elif self.prior == "gaussian":
            b_shift = self.shift(x)
            kl_loss, b = sample_gaussian(b_shift, b_logscale)

        return kl_loss, b

class VIEncoderLISTA(nn.Module):

    def __init__(self, img_size, dict_size, zeta, prior):
        super(VIEncoderLISTA, self).__init__()
        self.VI = VIEncoder(img_size, dict_size, prior)
        self.lambda_ = torch.tensor(zeta)
        self.W = nn.Sequential(
                        nn.Linear(dict_size, dict_size * 2),
                        nn.ReLU(),
                        nn.Linear(dict_size * 2, dict_size),
                        nn.ReLU(),
                        nn.Linear(dict_size, dict_size))

    def forward(self, x, A, iters=5):
        # Sample from VI
        kl_loss, b = self.VI(x)

        # Refine estimate with LISTA
        b_refine = soft_threshold(b, self.lambda_)
        for i in range(iters):
            b_refine = soft_threshold(b + self.W(b_refine), self.lambda_)
            
        return kl_loss, b_refine
