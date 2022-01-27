# nshepperd on GitHub
# Source: https://gist.github.com/nshepperd/9c90a95c5b3e2e61e62cc93066010c56

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def estimate_rejection_stat(encoder, train_data, dictionary, train_args, solver_args, default_device, num_samples=100, quantile=0.9):
    original_count = solver_args.num_samples
    solver_args.sample_method = "avg"
    solver_args.num_samples = 1
    rejection_stat = np.zeros(len(train_data))

    with torch.no_grad():
        for i in range(train_data.shape[0] // train_args.batch_size):
            patches = train_data[i * train_args.batch_size:(i + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T
            patch_idx = np.arange(i * train_args.batch_size, (i + 1) * train_args.batch_size)
            patches_cu = torch.tensor(patches.T).float().to(default_device)
            dict_cu = torch.tensor(dictionary, device=default_device).float()

            sample_loss = np.zeros((num_samples, train_args.batch_size))
            for j in range(num_samples):
                _, recon_loss, kl_loss, _, _ = encoder(patches_cu, dict_cu, patch_idx)
                sample_loss[j] = (recon_loss + kl_loss).mean(dim=-1).detach().cpu().numpy()

            sample_loss = np.sort(sample_loss, axis=0)
            rejection_stat[patch_idx] = sample_loss[int(quantile * len(sample_loss))]

    solver_args.sample_method = "rejection"
    solver_args.num_samples = original_count

    return rejection_stat

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def conditional_gumbel(logits, D, k=1):
    """Outputs k samples of Y = logits + StandardGumbel(), such that the
    argmax is given by D (one hot vector).

    """
    # iid. exponential
    E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])
    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True)
    # partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    # Sampled gumbel-adjusted softmax
    adjusted = (D * (-torch.log(E) + torch.log(Z)) +
                (1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
    return adjusted.detach() + logits - logits.detach()

def exact_conditional_gumbel(logits, D, k=1):
    """Same as conditional_gumbel but uses rejection sampling."""
    # Rejection sampling.
    idx = D.argmax(dim=-1)
    gumbels = []
    while len(gumbels) < k:
        gumbel = torch.rand_like(logits).log().neg().log().neg()
        if logits.add(gumbel).argmax() == idx:
            gumbels.append(logits+gumbel)
    return torch.stack(gumbels)

def gumbel_rao_argmax(logits, k, temp=1.0):
    """
    Returns the argmax(input, dim=-1) as a one-hot vector, with
    gumbel-rao gradient.

    k: integer number of samples to use in the rao-blackwellization.
    1 sample reduces to straight-through gumbel-softmax
    """
    num_classes = logits.shape[-1]
    I = torch.distributions.categorical.Categorical(logits=logits).sample()
    D = torch.nn.functional.one_hot(I, num_classes).float()
    adjusted = conditional_gumbel(logits, D, k=k)
    substitute = torch.nn.functional.softmax(adjusted/temp, dim=-1).mean(dim=0)
    return D.detach() + substitute - substitute.detach()

def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - zeta) * torch.sign(c)

def compute_loss(c, x0, x1, psi):
    T = (psi[None, :, :, :] * c[:, :, None, None]).sum(dim=1).reshape((
        x0.shape[0], psi.shape[1], psi.shape[2]))
    x1_hat = torch.matrix_exp(T) @ x0
    loss = F.mse_loss(x1_hat, x1, reduction='sum')
    return loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def FISTA_pytorch(x, A, dict_size, lambda_, max_iter=800, tol=1e-5, clip_grad=False, device='cpu'):
    z = nn.Parameter(torch.mul(torch.randn((len(x), dict_size), device=device),
                     0.3), requires_grad=True)
    z_opt = torch.optim.SGD([z], lr=1e-3, nesterov=True, momentum=0.9)
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(z_opt, gamma=0.995)
    change = 1e99
    k = 0
    while k < max_iter and change > tol:
        old_coeff = z.clone()

        z_opt.zero_grad()
        x_hat = A(z)
        loss = F.mse_loss(x_hat, x, reduction='none')
        loss.sum(dim=0).mean().backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_([z], 1e3)
        z_opt.step()
        opt_scheduler.step()

        with torch.no_grad():
            z.data = soft_threshold(z, get_lr(z_opt)*lambda_)

        change = torch.norm(z.data - old_coeff) / (torch.norm(old_coeff) + 1e-9)
        k += 1
    return (loss.detach(), get_lr(z_opt), k), z.data