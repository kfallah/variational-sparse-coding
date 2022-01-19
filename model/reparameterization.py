import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.solvers import FISTA, ADMM
from model.util import gumbel_rao_argmax

def compute_kl(solver_args, **kwargs):
    if solver_args.prior_method == "fixed":
        kl_loss = fixed_kl(solver_args, **kwargs)
    elif solver_args.prior_method == "vamp":
        kl_loss = vamp_kl(solver_args, **kwargs)
    elif solver_args.prior_method == "clf":
        kl_loss = clf_kl(solver_args, **kwargs)
    elif solver_args.prior_method == "coreset":
        kl_loss = coreset_kl(solver_args, **kwargs)
    else:
        raise NotImplementedError

    if solver_args.threshold and solver_args.theshold_learn:
        kl_loss += solver_args.gamma_kl_weight * kwargs['encoder'].lambda_kl_loss.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)

    return kl_loss

def fixed_kl(solver_args, **params):
    # One way to combat posterior collapse is to reduce the prior spread
    # The other common way is to reduce the overall weighting on the KL term
    # We implement both methods and defer to the configuration to select which to use
    scale_prior = torch.tensor(solver_args.spread_prior)
    logscale_prior = torch.log(scale_prior)
    if solver_args.prior_distribution == "laplacian":
        scale = torch.exp(params['logscale'])
        kl_loss = (params['shift'].abs() / scale_prior) + logscale_prior - params['logscale'] - 1
        kl_loss += (scale / scale_prior) * (-(params['shift'].abs() / scale)).exp()
    elif solver_args.prior_distribution == "gaussian":
        kl_loss = -0.5 * (1 + params['logscale'] - logscale_prior)
        kl_loss += 0.5 * ((params['shift'] ** 2) + (params['logscale']).exp()) / scale_prior
    elif solver_args.prior_distribution == "concreteslab":
        # Gaussian slab
        slab_kl = -0.5 * params['spike'] * (1 + params['logscale'] - logscale_prior)
        slab_kl += 0.5 * params['spike'] * ((params['shift'] ** 2) + params['logscale'].exp()) / scale_prior
        
        # Laplacian slab
        #scale = torch.exp(params['logscale'])
        #slab_kl = params['spike_prior'] * (params['shift'].abs() / scale_prior) + logscale_prior - params['logscale'] - 1
        #slab_kl += params['spike_prior'] * (scale / scale_prior) * (-(params['shift'].abs() / scale)).exp()        
        
        spike_kl = ((1 - params['spike']) * torch.log((1 - params['spike']) / (1 - params['spike_prior']))) + \
                    (params['spike'] * torch.log(params['spike'] / params['spike_prior']))
        kl_loss = slab_kl + spike_kl
    else:
        raise NotImplementedError

    return kl_loss

def vamp_kl(solver_args, **params):
    pseudo_feat =  params['encoder'].enc( params['encoder'].pseudo_inputs)
    pseudo_shift, pseudo_logscale =  params['encoder'].shift(pseudo_feat),  params['encoder'].scale(pseudo_feat)

    if solver_args.prior_distribution == "laplacian":
        log_p_z = (-0.5 * pseudo_logscale - torch.abs(params['z'].unsqueeze(2) - pseudo_shift) / pseudo_logscale.exp())
        log_p_z = torch.logsumexp(log_p_z - np.log(solver_args.num_pseudo_inputs), dim=-2)
        log_q_z = (-0.5 * params['logscale'] - torch.abs(params['z'] - params['shift']) / params['logscale'].exp())
    elif solver_args.prior_distribution == "gaussian":
        log_p_z = -0.5 * (pseudo_logscale + torch.pow(params['z'].unsqueeze(2) - pseudo_shift, 2 ) / (torch.exp(pseudo_logscale) + 1e-6))
        log_p_z = torch.logsumexp(log_p_z - np.log(solver_args.num_pseudo_inputs), dim=-2)
        log_q_z = -0.5 * (params['logscale'] + torch.pow(params['z'] - params['shift'], 2 ) / (torch.exp(params['logscale']) + 1e-6))
    elif solver_args.prior_distribution == "concreteslab":
        pseudo_logspike = -F.relu(-params['encoder'].spike(pseudo_feat))
        pseudo_spike = torch.clamp(pseudo_logspike.exp(), 1e-6, 1.0 - 1e-6) 
        log_p_z = -0.5 * (pseudo_logscale + torch.pow(params['z'].unsqueeze(2) - pseudo_shift, 2 ) / torch.exp(pseudo_logscale))
        log_p_z = params['spike'] *  torch.logsumexp(log_p_z - torch.log(solver_args.num_pseudo_inputs / pseudo_spike), dim=-2)
        log_p_z += (1 - params['spike']) * torch.log((1 - pseudo_spike).mean(dim=-2))

        log_q_z = params['spike'] * -0.5 * (params['logscale'] + torch.pow(params['z'] - params['shift'], 2 ) / torch.exp(params['logscale']))
        log_q_z += params['spike']*params['logspike'] + (1 - params['spike']) * torch.log(1 - params['spike'])

        log_q_z +=  (pseudo_spike*torch.log(pseudo_spike/params['spike_prior']) + \
                    (1 - pseudo_spike)*torch.log((1 - pseudo_spike)/(1 - params['spike_prior']))).mean(dim=0).sum()
    else:
        raise NotImplementedError

    return log_q_z - log_p_z 

def clf_kl(solver_args, **params):
    pseudo_feat =  params['encoder'].enc(params['encoder'].pseudo_inputs)
    pseudo_shift, pseudo_logscale =  params['encoder'].shift(pseudo_feat),  params['encoder'].scale(pseudo_feat)
    pseudo_shift, pseudo_logscale = pseudo_shift[None, None], pseudo_logscale[None, None]
    shift, logscale = params['shift'][:, :, None], params['logscale'][:, :, None]
    clf_logit = params['encoder'].clf(params['x'][:, 0])
    selection = F.gumbel_softmax(clf_logit, tau=params['encoder'].clf_temp, hard=True).unsqueeze(1)

    if solver_args.prior_distribution == "laplacian":
        kl_loss = ((shift - pseudo_shift).abs() / pseudo_logscale.exp()) + pseudo_logscale - logscale - 1
        kl_loss += (logscale.exp() / pseudo_logscale.exp()) * (-(shift - pseudo_shift).abs() / logscale.exp()).exp()
        kl_loss = (selection[..., None] * kl_loss).sum(dim=-2)
    elif solver_args.prior_distribution == "gaussian":
        kl_loss = -0.5 * (logscale - pseudo_logscale + 1)
        kl_loss += 0.5 * (((shift - pseudo_shift) ** 2) + logscale.exp()) / pseudo_logscale.exp()
        kl_loss = (selection[..., None] * kl_loss).sum(dim=-2)
    elif solver_args.prior_distribution == "concreteslab":
        spike = params['spike'][:, :, None]
        pseudo_logspike = -F.relu(-params['encoder'].spike(pseudo_feat))
        pseudo_spike = torch.clamp(pseudo_logspike.exp(), 1e-6, 1.0 - 1e-6) 

        slab_kl = -0.5 * spike * (1 + logscale - pseudo_logscale)
        slab_kl += 0.5 * spike * ((shift - pseudo_shift)**2 + logscale.exp()) / pseudo_logscale.exp()
        spike_kl = ((1 - spike) * torch.log((1 - spike) / (1 - pseudo_spike))) + \
                    (spike * torch.log(spike / pseudo_spike))
        kl_loss = slab_kl + spike_kl
        kl_loss = (selection[..., None] * kl_loss).sum(dim=-2)

        avg_pseudo_spike = pseudo_spike.mean(dim=-1)
        pseudo_prior = pseudo_spike.shape[1] * (avg_pseudo_spike*torch.log(avg_pseudo_spike/params['spike_prior']) + 
                            (1 - avg_pseudo_spike)*torch.log((1 - avg_pseudo_spike)/(1 - params['spike_prior'])))
        kl_loss += pseudo_prior.mean()
    else:
        raise NotImplementedError

    return kl_loss

def coreset_kl(solver_args, **params):
    if solver_args.coreset_weights == "encode":
        coreset_feat =  params['encoder'].enc(params['encoder'].coreset.float())
        pseudo_shift, pseudo_logscale =  params['encoder'].shift(coreset_feat),  params['encoder'].scale(coreset_feat)
    elif solver_args.coreset_weights == "FISTA":
        coreset_coeff = params['encoder'].coreset_coeff
        pseudo_shift = coreset_coeff
        pseudo_logscale = torch.zeros_like(pseudo_shift)
        pseudo_logscale[coreset_coeff.nonzero(as_tuple=True)] += 1.6
    else:
        raise NotImplementedError

    if solver_args.prior_distribution == "gaussian":
        if solver_args.coreset_prior_type == "mixture":
            pseudo_shift, pseudo_logscale = pseudo_shift[None, None], pseudo_logscale[None, None]
            log_p_z = -0.5 * (pseudo_logscale + torch.pow(params['z'].unsqueeze(2) - pseudo_shift, 2 ) / (torch.exp(pseudo_logscale) + 1e-6))
            log_p_z = torch.logsumexp(log_p_z - np.log(len(params['encoder'].coreset)), dim=-2)
            log_q_z = -0.5 * (params['logscale'] + torch.pow(params['z'] - params['shift'], 2 ) / (torch.exp(params['logscale']) + 1e-6))
            kl_loss = log_q_z - log_p_z 
        elif solver_args.coreset_prior_type == "single":
            kl_loss = -0.5 * (params['logscale'] - pseudo_logscale + 1)
            kl_loss += 0.5 * (((params['shift'] - pseudo_shift) ** 2) + params['logscale'].exp()) / pseudo_logscale.exp()
            coreset_idx = params['encoder'].coreset_labels[params['idx']]
            selection = F.one_hot(coreset_idx, len(params['encoder'].coreset))
            kl_loss = (selection[..., None] * kl_loss).sum(dim=-2)
    else:
        raise NotImplementedError

    return kl_loss

def sample_gaussian(shift, logscale, x, A, encoder, solver_args, idx=None):
    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, *torch.ones(x.dim(), dtype=int)).transpose(1, 0)
    shift = shift.repeat(solver_args.num_samples, *torch.ones(shift.dim(), dtype=int)).transpose(1, 0)
    logscale = logscale.repeat(solver_args.num_samples, *torch.ones(logscale.dim(), dtype=int)).transpose(1, 0)

    scale = torch.exp(0.5*logscale)
    eps = torch.randn_like(scale)
    z = shift + eps*scale

    if solver_args.threshold:
        if solver_args.estimator == "straight":
            z_thresh = encoder.soft_threshold(eps*scale.detach())
            non_zero = torch.nonzero(z_thresh, as_tuple=True)  
            z_thresh[non_zero] = shift[non_zero] + z_thresh[non_zero]
            z = z + z_thresh - z.detach()
        else:
            z_thresh = encoder.soft_threshold(eps*scale)
            non_zero = torch.nonzero(z_thresh, as_tuple=True)  
            z_thresh[non_zero] = shift[non_zero] + z_thresh[non_zero]
            z = z_thresh

    kl_loss = compute_kl(solver_args, x=x, z=(shift + eps*scale), encoder=encoder, 
                         logscale=logscale, shift=shift, idx=idx)
    recon_loss = compute_recon_loss(x, z, A)
    weight, iwae_loss = compute_loss(z, recon_loss.mean(dim=-1), 
                                     solver_args.kl_weight * kl_loss.sum(dim=-1), solver_args.sample_method)

    return iwae_loss, recon_loss.mean(dim=1), kl_loss.mean(dim=1), z, weight

def sample_laplacian(shift, logscale, x, A, encoder, solver_args, idx=None):
    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, *torch.ones(x.dim(), dtype=int)).transpose(1, 0)
    shift = shift.repeat(solver_args.num_samples, *torch.ones(shift.dim(), dtype=int)).transpose(1, 0)
    logscale = logscale.repeat(solver_args.num_samples, *torch.ones(logscale.dim(), dtype=int)).transpose(1, 0)

    scale = torch.exp(logscale)
    u = torch.rand_like(logscale) - 0.5
    eps = -scale * torch.sign(u) * torch.log((1.0-2.0*torch.abs(u)).clamp(min=1e-6)) 
    z = shift + eps * encoder.warmup
    if solver_args.threshold:
        if solver_args.estimator == "straight":
            z_thresh = encoder.soft_threshold(eps.detach() * encoder.warmup)
            non_zero = torch.nonzero(z_thresh, as_tuple=True)  
            z_thresh[non_zero] = shift[non_zero] + z_thresh[non_zero]
            z = z + z_thresh - z.detach()
        else:
            z_thresh = encoder.soft_threshold(eps*scale)
            non_zero = torch.nonzero(z_thresh, as_tuple=True)  
            z_thresh[non_zero] = shift[non_zero] + z_thresh[non_zero]
            z = z_thresh
    
    kl_loss = compute_kl(solver_args, x=x, z=(shift + eps), encoder=encoder, 
                         logscale=logscale, shift=shift, idx=idx)
    recon_loss = compute_recon_loss(x, z, A)
    weight, iwae_loss = compute_loss(z, recon_loss.mean(dim=-1), 
                                     solver_args.kl_weight * kl_loss.sum(dim=-1), solver_args.sample_method)

    return iwae_loss, recon_loss.mean(dim=1), kl_loss.mean(dim=1), z, weight

def sample_concreteslab(shift, logscale, logspike, x, A, encoder, solver_args, temp=0.1, spike_prior=0.2, idx=None):
    # From first submission of (Tonolini et al 2020) without pseudo-inputs

    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, *torch.ones(x.dim(), dtype=int)).transpose(1, 0)
    shift = shift.repeat(solver_args.num_samples, *torch.ones(shift.dim(), dtype=int)).transpose(1, 0)
    logscale = logscale.repeat(solver_args.num_samples, *torch.ones(logscale.dim(), dtype=int)).transpose(1, 0)
    logspike = logspike.repeat(solver_args.num_samples, *torch.ones(logspike.dim(), dtype=int)).transpose(1, 0)
    spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 

    # Gaussian slab
    std = encoder.warmup * torch.exp(0.5*logscale) + np.sqrt(1 - encoder.warmup)
    eps = torch.randn_like(std)
    slab = eps.mul(std) + (encoder.warmup * shift)

    # Laplacian slab
    #std = encoder.warmup * torch.exp(logscale) + (1 - encoder.warmup)
    #u = torch.rand_like(logscale) - 0.5
    #eps = -std * torch.sign(u) * torch.log((1.0-2.0*torch.abs(u)).clamp(min=1e-6)) 
    #slab = (encoder.warmup * shift) + eps
    
    eta = torch.rand_like(std)
    u = torch.log(eta) - torch.log(1 - eta)
    selection = torch.sigmoid((u + logspike) / temp)

    if solver_args.estimator == "gumbel":
        selection_use = selection
    elif solver_args.estimator == "straight":
        selection_passthru = torch.round(selection)
        selection_use = selection + (selection_passthru - selection).detach()
    elif solver_args.estimator == "gumbelrao":
        spike_logit = torch.stack([(1 - spike), spike], dim=-1)
        selection = gumbel_rao_argmax(spike_logit, 20, temp=temp)
        selection_use = torch.argmax(selection, dim=-1)

    z = selection_use * slab

    kl_loss = compute_kl(solver_args, x=x, z=z, encoder=encoder, logscale=logscale,
                         shift=shift, logspike=logspike, spike=spike, 
                         spike_prior=spike_prior, idx=idx)
    recon_loss = compute_recon_loss(x, z, A)
    weight, iwae_loss = compute_loss(z, recon_loss.mean(dim=-1), 
                                     solver_args.kl_weight * kl_loss.sum(dim=-1), solver_args.sample_method)

    return iwae_loss, recon_loss.mean(dim=1), kl_loss.mean(dim=1), z, weight

def compute_recon_loss(x, z, A):
    if issubclass(type(A), nn.Module):
        x_hat = A(z)
    else:
        x_hat = (z @ A.T)
    recon_loss = F.mse_loss(x_hat, x, reduction='none').reshape(len(x), x.shape[1], -1)

    return recon_loss

def compute_loss(z, recon_loss, kl_loss, sample_method="avg"):
    log_loss = recon_loss + kl_loss

    if sample_method == "iwae":
        # Compute importance weights through softmax (due to computing log prob)
        weight = F.softmax(-log_loss, dim=-1)
        # Compute weighted sum
        iwae_loss = (weight * log_loss).sum(dim=-1).mean()
    elif sample_method == "max":
        # Apply our sampling procedure which takes the sample with lowest ELBO, encouraging
        # feature reuse
        z_idx = torch.argmin(log_loss, dim=-1).detach()
        weight = torch.zeros_like(recon_loss)
        weight[torch.arange(len(z)), z_idx] = 1
    else:
        # Apply standard sampling from (Kingma & Welling)
        weight = torch.ones_like(recon_loss) / recon_loss.shape[-1]

    iwae_loss = (weight * log_loss).sum(dim=-1).mean()
    return weight, iwae_loss