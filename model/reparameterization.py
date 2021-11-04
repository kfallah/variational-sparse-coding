import logging

import torch
import torch.nn.functional as F

# TODO: need to fix sampling in case of convolutional architecture
def sample_gaussian(mu, logvar, x, A, solver_args):
    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    mu = mu.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logvar = logvar.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)

    scale = torch.exp(0.5*logvar)
    eps = torch.randn_like(scale)
    z = mu + eps*scale

    recon_loss = F.mse_loss((z @ A.T), x, reduction='none').mean(dim=-1)
    kl_loss = - solver_args.kl_weight * 0.5 * (1 + logvar - (mu ** 2) - logvar.exp()).mean(dim=-1)

    z, iwae_loss = compute_iwae_loss(z, recon_loss, kl_loss, solver_args.iwae)

    return iwae_loss, recon_loss.mean(), kl_loss.mean(), z

def sample_laplacian(mu, logscale, x, A, solver_args):
    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    mu = mu.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logscale = logscale.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)

    scale = torch.exp(logscale)
    u = torch.rand_like(logscale) - 0.5
    z = mu - scale * torch.sign(u) * torch.log((1.0-2.0*torch.abs(u)).clamp(min=1e-10))   
    
    recon_loss = F.mse_loss((z @ A.T), x, reduction='none').mean(dim=-1)
    kl_loss = solver_args.kl_weight * (-logscale + mu.abs() + scale*(-mu.abs() / scale).exp()).mean(dim=-1)
    
    z, iwae_loss = compute_iwae_loss(z, recon_loss, kl_loss, solver_args.iwae)

    return iwae_loss, recon_loss.mean(), kl_loss.mean(), z

def sample_spikeslab(mu, logscale, logspike, x, A, solver_args, c=50, alpha=0.2):
    # From first submission of (Tonolini et al 2020) without pseudo-inputs
    # Code found https://github.com/Alfo5123/Variational-Sparse-Coding/blob/master/src/models/vsc.py

    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    mu = mu.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logscale = logscale.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logspike = logspike.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)

    std = torch.exp(0.5*logscale)
    eps = torch.randn_like(std)
    gaussian = eps.mul(std).add_(mu)
    eta = torch.rand_like(std)
    selection = torch.sigmoid(c*(eta + logspike.exp() - 1))
    z = selection * gaussian

    spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 
    recon_loss = F.mse_loss((z @ A.T), x, reduction='none').mean(dim=-1)
    kl_loss = -0.5 * solver_args.slab_kl_weight * (spike * (1 + logscale - mu.pow(2) - logscale.exp())).sum(dim=-1) + \
               solver_args.spike_kl_weight * ((1 - spike) * (torch.log((1 - spike) / (1 - alpha))) + \
                                              (spike * (torch.log(spike / alpha)))).sum(dim=-1)

    z, iwae_loss = compute_iwae_loss(z, recon_loss, kl_loss, solver_args.iwae)

    return iwae_loss, recon_loss.mean(), kl_loss.mean(), z

def sample_concreteslab(mu, logscale, logspike, x, A, solver_args, temp=0.1, alpha=0.2):
    # From first submission of (Tonolini et al 2020) without pseudo-inputs
    # Code found https://github.com/Alfo5123/Variational-Sparse-Coding/blob/master/src/models/vsc.py

    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    mu = mu.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logscale = logscale.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logspike = logspike.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)

    std = torch.exp(0.5*logscale)
    eps = torch.randn_like(std)
    gaussian = eps.mul(std).add_(mu)
    eta = torch.rand_like(std)
    u = torch.log(eta) - torch.log(1 - eta)
    selection = torch.sigmoid((u + logspike) / temp)
    z = selection * gaussian

    spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 
    recon_loss = F.mse_loss((z @ A.T), x, reduction='none').mean(dim=-1)
    kl_loss = -0.5 * solver_args.slab_kl_weight * (spike.mul(1 + logscale - mu.pow(2) - logscale.exp())).sum(dim=-1) + \
               solver_args.spike_kl_weight * ((1 - spike).mul(torch.log((1 - spike) / (1 - alpha))) + \
                                              spike.mul(torch.log(spike / alpha))).sum(dim=-1)

    z, iwae_loss = compute_iwae_loss(z, recon_loss, kl_loss, solver_args.iwae)

    return iwae_loss, recon_loss.mean(), kl_loss.mean(), z

def compute_iwae_loss(z, recon_loss, kl_loss, iwae=False):
    if iwae:
        # Compute importance weights through softmax (due to computing log prob)
        log_loss = recon_loss + kl_loss
        weight = F.softmax(log_loss, dim=-1)
        # Take z with largest weight
        z_idx = torch.argmax(weight, dim=-1).detach()
        z = z[torch.arange(len(z)), z_idx]
        # Compute weighted sum
        iwae_loss = (weight * log_loss).sum(dim=-1).mean()
    else:
        # In traditional VAE, the loss is just a simple average over samples
        z = z[:, 0]
        iwae_loss = (recon_loss + kl_loss).mean()
    return z, iwae_loss