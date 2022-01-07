# nshepperd on GitHub
# Source: https://gist.github.com/nshepperd/9c90a95c5b3e2e61e62cc93066010c56

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def FISTA_pytorch(x, A, dict_size, lambda_, max_iter=800, tol=1e-5, device='cpu'):
    z = nn.Parameter(torch.mul(torch.randn((len(x), dict_size), device=device),
                     0.02), requires_grad=True)
    z_opt = torch.optim.SGD([z], lr=1e-2, nesterov=True, momentum=0.9)
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(z_opt, gamma=0.985)
    change = 1e99
    k = 0
    while k < max_iter and change > tol:
        old_coeff = z.clone()

        z_opt.zero_grad()
        x_hat = A(z)
        loss = F.mse_loss(x_hat, x, reduction='sum')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], 1e2)
        z_opt.step()
        opt_scheduler.step()

        with torch.no_grad():
            z.data = soft_threshold(z, get_lr(z_opt)*lambda_)

        change = torch.norm(z.data - old_coeff) / (torch.norm(old_coeff) + 1e-9)
        k += 1
    return (loss.item(), get_lr(z_opt), k), z.data