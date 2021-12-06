import logging

import numpy as np

import torch
import torch.nn.functional as F

def compute_kl(solver_args, **kwargs):
    if solver_args.prior_method == "fixed":
        return fixed_kl(solver_args, **kwargs)
    elif solver_args.prior_method == "vamp":
        return vamp_kl(solver_args, **kwargs)
    elif solver_args.prior_method == "clf":
        return clf_kl(solver_args, **kwargs)
    elif solver_args.prior_method == "coreset":
        return coreset_kl(solver_args, **kwargs)
    else:
        raise NotImplementedError

def fixed_kl(solver_args, **params):
    # One way to combat posterior collapse is to reduce the prior spread
    # The other common way is to reduce the overall weighting on the KL term
    # We implement both methods and defer to the configuration to select which to use
    scale_prior = torch.tensor(solver_args.spread_prior)
    logscale_prior = torch.log(scale_prior)
    if solver_args.prior_distribution == "laplacian":
        scale = torch.exp(params['logscale'])
        kl_loss = (params['shift'].abs() / scale_prior) + logscale_prior - params['logscale'] - 1
        kl_loss += (scale / scale_prior) * (-params['shift'].abs() / scale).exp()
    elif solver_args.prior_distribution == "gaussian":
        kl_loss = -0.5 * (1 + params['logscale'] - logscale_prior)
        kl_loss += 0.5 * ((params['shift'] ** 2) + (params['logscale']).exp()) / scale_prior
    elif solver_args.prior_distribution == "concreteslab":
        slab_kl = -0.5 * params['spike'] * (1 + params['logscale'] - logscale_prior)
        slab_kl += 0.5 * params['spike'] * ((params['shift'] ** 2) + params['logscale'].exp()) / scale_prior
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
        log_p_z = (-0.5 * pseudo_logscale - torch.abs(params['z'].unsqueeze(2) - pseudo_shift) / pseudo_logscale.exp()).sum(dim=-1)
        log_p_z = torch.logsumexp(log_p_z - np.log(solver_args.num_pseudo_inputs), dim=-2)
        log_q_z = (-0.5 * params['logscale'] - torch.abs(params['z'] - params['shift']) / params['logscale'].exp())
    elif solver_args.prior_distribution == "gaussian":
        log_p_z = -0.5 * (pseudo_logscale + torch.pow(params['z'].unsqueeze(2) - pseudo_shift, 2 ) / torch.exp(pseudo_logscale))
        log_p_z = torch.logsumexp(log_p_z - np.log(solver_args.num_pseudo_inputs), dim=-2)
        log_q_z = -0.5 * (params['logscale'] + torch.pow(params['z'] - params['shift'], 2 ) / torch.exp(params['logscale']))
    elif solver_args.prior_distribution == "concreteslab":
        pseudo_logspike = -F.relu(-params['encoder'].spike(pseudo_feat))
        pseudo_spike = torch.clamp(pseudo_logspike.exp(), 1e-6, 1.0 - 1e-6) 
        log_p_z = -0.5 * (pseudo_logscale + torch.pow(params['z'].unsqueeze(2) - pseudo_shift, 2 ) / torch.exp(pseudo_logscale))
        log_p_z = params['spike'] *  torch.logsumexp(log_p_z - torch.log(solver_args.num_pseudo_inputs / pseudo_spike), dim=-2)
        log_p_z += (1 - params['spike']) * torch.log((1 - pseudo_spike).mean(dim=-2))
        log_q_z = params['spike'] * -0.5 * (params['logscale'] + torch.pow(params['z'] - params['shift'], 2 ) / torch.exp(params['logscale']))
        log_q_z += (1 - params['spike']) * torch.log(1 - params['spike'])
    else:
        raise NotImplementedError

    return log_q_z - log_p_z

def clf_kl(solver_args, **params):
    pseudo_feat =  params['encoder'].enc(params['encoder'].pseudo_inputs)
    pseudo_shift, pseudo_logscale =  params['encoder'].shift(pseudo_feat),  params['encoder'].scale(pseudo_feat)
    pseudo_shift, pseudo_logscale = pseudo_shift[None, None], pseudo_logscale[None, None]
    shift, logscale = params['shift'][:, :, None], params['logscale'][:, :, None]
    clf_logit = params['encoder'].clf(params['x'])
    selection = F.gumbel_softmax(clf_logit, tau=params['encoder'].clf_temp)

    if solver_args.prior_distribution == "laplacian":
        kl_loss = ((shift - pseudo_shift).abs() / pseudo_logscale.exp()) + pseudo_logscale - logscale - 1
        kl_loss += (logscale.exp() / pseudo_logscale.exp()) * (-(shift - pseudo_shift).abs() / logscale.exp()).exp()
        kl_loss = (selection[..., None] * kl_loss).sum(dim=-2)
    elif solver_args.prior_distribution == "gaussian":
        kl_loss = -0.5 * (logscale - pseudo_logscale + 1)
        kl_loss += 0.5 * (((shift - pseudo_shift) ** 2) + logscale.exp()) / pseudo_logscale.exp()
        kl_loss = (selection[..., None] * kl_loss).sum(dim=-2)
    elif solver_args.prior_distribution == "concreteslab":
        raise NotImplementedError

    return kl_loss

def coreset_kl():
    raise NotImplementedError

# TODO: need to fix sampling in case of convolutional architecture
def sample_gaussian(shift, logscale, x, A, encoder, solver_args):
    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    shift = shift.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logscale = logscale.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)

    scale = torch.exp(0.5*logscale)
    eps = torch.randn_like(scale)
    if solver_args.threshold:
        z = encoder.soft_threshold(eps*scale)
        non_zero = torch.nonzero(z, as_tuple=True)  
        z[non_zero] = shift[non_zero] + z[non_zero]
    else:
        z = shift + eps*scale

    kl_loss = compute_kl(solver_args, x=x, z=z, encoder=encoder, logscale=logscale, shift=shift).sum(dim=-1)
    #if solver_args.threshold:
    #    z = encoder.soft_threshold(z)
    recon_loss = F.mse_loss((z @ A.T), x, reduction='none').mean(dim=-1)
    z, iwae_loss = compute_iwae_loss(z, recon_loss, solver_args.kl_weight * kl_loss, solver_args.iwae)

    return iwae_loss, recon_loss.mean(), kl_loss.mean(), z

def sample_laplacian(shift, logscale, x, A, encoder, solver_args):
    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    shift = shift.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logscale = logscale.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)

    scale = torch.exp(logscale)
    u = torch.rand_like(logscale) - 0.5
    z = shift - scale * torch.sign(u) * torch.log((1.0-2.0*torch.abs(u)).clamp(min=1e-10))   

    kl_loss = compute_kl(solver_args, x=x, z=z, encoeder=encoder, logscale=logscale, shift=shift).sum(dim=-1)
    if solver_args.threshold:
        z = encoder.soft_threshold(z)
    recon_loss = F.mse_loss((z @ A.T), x, reduction='none').mean(dim=-1)    
    z, iwae_loss = compute_iwae_loss(z, recon_loss, solver_args.kl_weight * kl_loss, solver_args.iwae)

    return iwae_loss, recon_loss.mean(), kl_loss.mean(), z

def sample_concreteslab(shift, logscale, logspike, x, A, encoder, solver_args, temp=0.1, spike_prior=0.2):
    # From first submission of (Tonolini et al 2020) without pseudo-inputs
    # The difference with the spike-slab is the use of the parameterization from the Concrete distribution paper
    # These seem to be the exact same, but this parameterization approach lets us compare to the hyper-parameters
    # Used in the concrete distribution

    # Repeat based on the number of samples
    x = x.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    shift = shift.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logscale = logscale.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)
    logspike = logspike.repeat(solver_args.num_samples, 1, 1).permute(1, 0, 2)

    std = torch.exp(0.5*logscale)
    eps = torch.randn_like(std)
    gaussian = eps.mul(std).add_(shift)
    eta = torch.rand_like(std)
    u = torch.log(eta) - torch.log(1 - eta)
    selection = torch.sigmoid((u + logspike) / temp)
    z = selection * gaussian

    spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 
    kl_loss = compute_kl(solver_args, x=x, z=z, encoder=encoder, logscale=logscale,
                         shift=shift, spike=spike, spike_prior=spike_prior).sum(dim=-1)
    if solver_args.threshold:
        z = encoder.soft_threshold(z)
    recon_loss = F.mse_loss((z @ A.T), x, reduction='none').mean(dim=-1)
    z, iwae_loss = compute_iwae_loss(z, recon_loss, solver_args.kl_weight * kl_loss, solver_args.iwae)

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