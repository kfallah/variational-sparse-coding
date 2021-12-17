# nshepperd on GitHub
# Source: https://gist.github.com/nshepperd/9c90a95c5b3e2e61e62cc93066010c56

import torch

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
