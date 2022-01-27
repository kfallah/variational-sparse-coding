import sys
import logging
sys.path.append('.')

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.distributions import gamma as gamma

from model.reparameterization import *

from model.feature_enc import MLPEncoder, ConvEncoder

class VIEncoder(nn.Module):

    def __init__(self, img_size, dict_size, solver_args):
        super(VIEncoder, self).__init__()
        self.solver_args = solver_args
        self.dict_size = dict_size

        if self.solver_args.feature_enc == "MLP":
            self.enc = MLPEncoder(int(np.sqrt(img_size)))
            input_size = (img_size**2)
        elif self.solver_args.feature_enc == "CONV":
            img_size = 256
            self.enc = ConvEncoder(img_size, 3)
            input_size = (3, 64, 64)
        elif self.solver_args.feature_enc == "RES":
            self.enc = models.resnet18()
            self.enc.fc = nn.Identity()
            img_size = 512
            input_size = (3, 64, 64)
        else:
            raise NotImplementedError

        self.scale = nn.Linear(img_size, dict_size)
        self.shift = nn.Linear(img_size, dict_size)

        if self.solver_args.prior_distribution == "concreteslab":
            self.spike = nn.Linear(img_size, dict_size)
            self.temp = 1.0
            self.warmup = 0.1
        if self.solver_args.threshold and self.solver_args.theshold_learn:
            self.lambda_prior_alpha = nn.Linear(img_size, dict_size)
            self.lambda_prior_beta = nn.Linear(img_size, dict_size)
        if self.solver_args.prior_distribution == "laplacian":
            self.warmup = 0.1

        if self.solver_args.prior_method == "vamp" or self.solver_args.prior_method == "clf":
            pseudo_init = torch.randn(self.solver_args.num_pseudo_inputs, *input_size)
            self.pseudo_inputs = nn.Parameter(pseudo_init, requires_grad=True)
        if self.solver_args.prior_method == "clf":
            self.clf_temp = 1.0
            if self.solver_args.feature_enc == "MLP":
                self.clf = nn.Sequential(
                            MLPEncoder(int(np.sqrt(img_size))),
                            nn.Linear(img_size, self.solver_args.num_pseudo_inputs)
                        )
            else:
                self.clf = models.resnet18() 
                self.clf.fc = nn.Linear(512, self.solver_args.num_pseudo_inputs)

    def ramp_hyperparams(self):
        self.temp = 1e-2
        self.clf_temp = 1e-2
        self.warmup = 1.0

    def soft_threshold(self, z):
        return F.relu(torch.abs(z) - torch.abs(self.lambda_)) * torch.sign(z)

    def forward(self, x, decoder, idx=None):
        feat = self.enc(x)
        b_logscale = self.scale(feat)
        b_shift = self.shift(feat)

        if self.solver_args.threshold:
            if self.solver_args.theshold_learn:
                alpha = self.lambda_prior_alpha(feat).exp().clip(1e-6, 1e6)
                beta = self.lambda_prior_beta(feat).exp().clip(1e-6, 1e6)
                gamma_pred = gamma.Gamma(alpha, beta)
                gamma_prior = gamma.Gamma(3, (3 * torch.ones_like(beta)) / self.solver_args.threshold_lambda)

                self.lambda_ = gamma_pred.rsample([self.solver_args.num_samples]).transpose(1, 0)
                self.lambda_kl_loss = torch.distributions.kl.kl_divergence(gamma_pred, gamma_prior)
            else:
                self.lambda_ = torch.ones_like(b_logscale) * self.solver_args.threshold_lambda
                self.lambda_ = self.lambda_.repeat(self.solver_args.num_samples, 
                                                   *torch.ones(self.lambda_.dim(), dtype=int)).transpose(1, 0)


        if self.solver_args.prior_distribution == "laplacian":
            iwae_loss, recon_loss, kl_loss, sparse_codes, weight = sample_laplacian(b_shift, b_logscale, x, decoder,
                                                                 self, self.solver_args, idx=idx)
        elif self.solver_args.prior_distribution == "gaussian":
            iwae_loss, recon_loss, kl_loss, sparse_codes, weight  = sample_gaussian(b_shift, b_logscale, x, decoder,
                                                                 self, self.solver_args, idx=idx)
        elif self.solver_args.prior_distribution == "concreteslab":
            logspike = -F.relu(-self.spike(feat))
            iwae_loss, recon_loss, kl_loss, sparse_codes, weight = sample_concreteslab(b_shift, b_logscale, logspike, x, decoder, 
                                                                    self, self.solver_args, 
                                                                    self.temp, self.solver_args.spike_prior,
                                                                    idx=idx)       
        else:
            raise NotImplementedError
        
        return iwae_loss, recon_loss, kl_loss, sparse_codes, weight