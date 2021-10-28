import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.kl as kl


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

def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - torch.abs(zeta)) * torch.sign(c)
    
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

class VIEncoder(nn.Module):

    def __init__(self, img_size, dict_size, prior):
        super(VIEncoder, self).__init__()
        self.prior = prior
        self.enc = MLPEncoder(img_size)
        self.scale = nn.Linear((img_size**2), dict_size)
        self.shift = nn.Linear((img_size**2), dict_size)

    def forward(self, x, A):
        feat = self.enc(x)
        b_logscale = self.scale(feat)
        b_shift = self.shift(feat)

        if self.prior == "laplacian":
            kl_loss, b = sample_laplacian(b_shift, b_logscale)
        elif self.prior == "gaussian":
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

class LISTA(nn.Module):
    
    def __init__(self, img_size, dict_size, zeta):
        super(LISTA, self).__init__()

        self.lambda_ = torch.tensor(zeta)
        #self.lambda_ = nn.Parameter(torch.tensor(zeta), requires_grad=True)
        self.W = nn.Sequential(
                        nn.Linear(img_size**2, (img_size**2) * 2),
                        nn.ReLU(),
                        nn.Linear((img_size**2) * 2, (img_size**2)),
                        nn.ReLU(),
                        nn.Linear((img_size**2), dict_size))
        self.S = nn.Sequential(
                        nn.Linear(dict_size, dict_size * 2),
                        nn.ReLU(),
                        nn.Linear(dict_size * 2, dict_size),
                        nn.ReLU(),
                        nn.Linear(dict_size, dict_size))

    def forward(self, x, iters=5):
        B = self.W(x)
        Z = soft_threshold(B, self.lambda_)
        for i in range(iters):
            Z = soft_threshold(B + self.S(Z), self.lambda_)
        return 0., Z
