import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vi_encoder import VIEncoder

def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - torch.abs(zeta)) * torch.sign(c)

class VIEncoderLISTA(nn.Module):

    def __init__(self, img_size, dict_size, zeta, solver_args):
        super(VIEncoderLISTA, self).__init__()
        self.VI = VIEncoder(img_size, dict_size, solver_args)
        self.lambda_ = torch.tensor(zeta)
        self.W = nn.Sequential(
                        nn.Linear(dict_size, dict_size * 2),
                        nn.ReLU(),
                        nn.Linear(dict_size * 2, dict_size),
                        nn.ReLU(),
                        nn.Linear(dict_size, dict_size))

    def forward(self, x, A, iters=5):
        # Sample from VI
        kl_loss, b = self.VI(x, A)

        # Refine estimate with LISTA
        b_refine = soft_threshold(b, self.lambda_)
        for i in range(iters):
            b_refine = soft_threshold(b + self.W(b_refine), self.lambda_)
            
        return kl_loss, b_refine

class LISTA(nn.Module):
    
    def __init__(self, img_size, dict_size, zeta):
        super(LISTA, self).__init__()

        #self.lambda_ = torch.tensor(0.5)
        self.lambda_ = nn.Parameter(torch.ones(dict_size)*1e-2, requires_grad=True)
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

        self.W = nn.Linear(img_size**2, dict_size, bias=False)
        self.S = nn.Linear(dict_size, dict_size, bias=False)

    def forward(self, x, iters=5):
        B = self.W(x)
        Z = soft_threshold(B, self.lambda_)
        for i in range(iters):
            Z = soft_threshold(B + self.S(Z), self.lambda_)
        return 0., Z
