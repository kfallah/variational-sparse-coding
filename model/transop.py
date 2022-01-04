import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransOp_expm(nn.Module):

    def __init__(self, M=6, N=3, var=1e-3):
        super(TransOp_expm, self).__init__()
        self.psi = nn.Parameter(torch.mul(torch.randn((M, N, N)), var), requires_grad=True)
        self.psi.data = self.psi.data / self.psi.reshape(M, -1).norm(dim=1)[:, None, None]
        self.M = M
        self.N = N
        self.latent_points = None

    def forward(self, x):
        T = torch.einsum('bm,mpk->bpk', self.c, self.psi)
        out = torch.matrix_exp(T) @ x
        return out

    def set_coefficients(self, c):
        self.c = c

    def get_psi(self):
        return self.psi.data

    def set_psi(self,psi_input):
        self.psi.data = psi_input