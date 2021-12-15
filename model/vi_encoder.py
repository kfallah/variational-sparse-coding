import sys
import logging
sys.path.append('.')

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model.reparameterization import *
from model.feature_enc import MLPEncoder

class VIEncoder(nn.Module):

    def __init__(self, img_size, dict_size, solver_args):
        super(VIEncoder, self).__init__()
        self.solver_args = solver_args
        if self.solver_args.feature_enc == "MLP":
            self.enc = MLPEncoder(img_size)
            if self.solver_args.prior_method == "clf":
                self.clf = nn.Sequential(
                            MLPEncoder(img_size),
                            nn.Linear((img_size**2), self.solver_args.num_pseudo_inputs)
                        )
        else:
            raise NotImplementedError
        self.scale = nn.Linear((img_size**2), dict_size)
        self.shift = nn.Linear((img_size**2), dict_size)

        if self.solver_args.prior_distribution == "concreteslab":
            self.spike = nn.Linear((img_size**2), dict_size)
            self.temp = 1.0
            self.warmup = 0.0
        
        if self.solver_args.prior_distribution == "laplacian":
            self.warmup = 0.1

        if self.solver_args.prior_method == "vamp" or self.solver_args.prior_method == "clf":
            pseudo_init = torch.randn(self.solver_args.num_pseudo_inputs, (img_size**2))
            self.pseudo_inputs = nn.Parameter(pseudo_init, requires_grad=True)
        if self.solver_args.prior_method == "clf":
           self.clf_temp = 1.0

        if self.solver_args.threshold:
            self.lambda_ = torch.ones(dict_size) * solver_args.threshold_lambda
            if self.solver_args.theshold_learn:
                self.lambda_ = nn.Parameter(self.lambda_, requires_grad=True)

    def soft_threshold(self, z):
        if not self.solver_args.theshold_learn:
            self.lambda_ = torch.ones(z.shape[-1], device=z.device) * self.solver_args.threshold_lambda
        return F.relu(torch.abs(z) - torch.abs(self.lambda_)) * torch.sign(z)

    def forward(self, x, A, idx=None):
        feat = self.enc(x)
        b_logscale = self.scale(feat)
        b_shift = self.shift(feat)

        if self.solver_args.prior_distribution == "laplacian":
            iwae_loss, recon_loss, kl_loss, b = sample_laplacian(b_shift, b_logscale, x, A,
                                                                 self, self.solver_args, idx=idx)
        elif self.solver_args.prior_distribution == "gaussian":
            iwae_loss, recon_loss, kl_loss, b  = sample_gaussian(b_shift, b_logscale, x, A,
                                                                 self, self.solver_args, idx=idx)
        elif self.solver_args.prior_distribution == "concreteslab":
            logspike = -F.relu(-self.spike(feat))
            iwae_loss, recon_loss, kl_loss, b = sample_concreteslab(b_shift, b_logscale, logspike, x, A, 
                                                                    self, self.solver_args, 
                                                                    self.temp, self.solver_args.spike_prior,
                                                                    idx=idx)       
        else:
            raise NotImplementedError
        
        return iwae_loss, recon_loss, kl_loss, b