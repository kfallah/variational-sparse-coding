# %%
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from model.vi_encoder import VIEncoder
from utils.solvers import FISTA
from utils.data_loader import load_whitened_images


# %%
# Coadapt baseline
file_list = [
    "prior_comp/concreteslab_gradvar/",
    "prior_comp/concreteslab_gumbel/",
    "prior_comp/concreteslab_raogumbel/",
    "prior_comp/gaussian_nothresh_iwae/",
    "prior_comp/gaussian_gradvar/",
    "prior_comp/laplacian_fixed_iwae/",
    "prior_comp/laplacian_thresh_iwae/",
]

file_labels = [
    "CS GumbelSoftmax Estimator",
    "CS StraightThrough Estimator",
    "CS GumbelRao Estimator",
    "Gaussian",
    "Thresholded Gaussian",
    "Laplacian",
    "Thresholded Laplacian"
]

base_lambda = 4.0

# %%
#with open(base_run + "config.json") as json_data:
with open(file_list[0] + 'config.json') as json_data:
    config_data = json.load(json_data)
train_args = SimpleNamespace(**config_data['train'])
gt_dictionary = np.load(file_list[0] + 'train_savefile.npz')['phi'][-1]

train_patches, val_patches = load_whitened_images(train_args, gt_dictionary)
p_signal = np.var(val_patches.reshape(len(val_patches), -1), axis=-1).mean()

# %%
grad_bias = {}
grad_var = {}

for idx, train_run in enumerate(file_list):
    print(f"Method {file_labels[idx]}")
    grad_bias[file_labels[idx]] = {}
    grad_var[file_labels[idx]] = {}

    with open(train_run + 'config.json') as json_data:
        config_data = json.load(json_data)
    train_args = SimpleNamespace(**config_data['train'])
    solver_args = SimpleNamespace(**config_data['solver'])

    epoch_list = [int(re.search(r'epoch([0-9].*).pt', f)[1]) for f in os.listdir(train_run) if re.search(r'epoch([0-9].*).pt', f)]
    epoch_list = np.sort(epoch_list)

    for epoch in epoch_list:
        grad_bias[file_labels[idx]][epoch] = 0
        grad_var[file_labels[idx]][epoch] = 0

        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)
        
        encoder = VIEncoder(train_args.patch_size, train_args.dict_size, solver_args).to('cuda:1')
        encoder.load_state_dict(torch.load(train_run + f"encoderstate_epoch{epoch}.pt")['model_state'])
        encoder.ramp_hyperparams()

        if epoch == 0:
            phi = np.random.randn(train_args.patch_size ** 2, train_args.dict_size)
            phi /= np.sqrt(np.sum(phi ** 2, axis=0))
            phi = torch.tensor(phi, device='cuda:1').float()
        else:
            phi = torch.tensor(np.load(train_run + 'train_savefile.npz')['phi'][epoch - 1], device='cuda:1').float()

        dict_cu = torch.nn.Parameter(phi.clone().detach(), requires_grad=True)
        sgd = torch.optim.SGD(encoder.parameters(), lr=1e0)
        
        for j in range(val_patches.shape[0] // train_args.batch_size):
            # Load next batch of validation patches
            patches = val_patches[j * train_args.batch_size:(j + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T
            patches_cu = torch.tensor(patches.T).float().to('cuda:1')

            encoder.solver_args.iwae = True
            encoder.solver_args.num_samples = 200
            sgd.zero_grad()
            iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu.detach()) 
            iwae_loss.backward()
            model_grad = [param.grad.data.reshape(-1).detach().cpu() for param in encoder.parameters()]
            model_grad = torch.cat(model_grad)
            true_grad = model_grad

            batch_var = []
            for k in range(100):
                encoder.solver_args.iwae = True
                encoder.solver_args.num_samples = 200
                sgd.zero_grad()
                iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu.detach()) 
                iwae_loss.backward()

                model_grad = [param.grad.data.reshape(-1).detach().cpu() for param in encoder.parameters()]
                model_grad = torch.cat(model_grad)

                batch_var.append(model_grad)

            batch_var = torch.stack(batch_var)
            grad_bias[file_labels[idx]][epoch] += torch.linalg.norm(true_grad - torch.mean(batch_var, dim=0)) / (val_patches.shape[0] // train_args.batch_size)
            grad_var[file_labels[idx]][epoch] += torch.var(batch_var, dim=0).mean() / (val_patches.shape[0] // train_args.batch_size)

        print(f"Epoch {epoch}, grad bias: {grad_bias[file_labels[idx]][epoch]:.3E}, grad var: {grad_var[file_labels[idx]][epoch]:.3E}")
    print()

np.savez_compressed("figures/grad_stats/vscgrad_save.npz",
        grad_bias=grad_bias, file_list=file_list, file_labels=file_labels)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

for idx, label in enumerate(file_labels):
    grad_list = sorted(grad_bias[label].items())
    x_grad, grad_plt = zip(*grad_list)
    grad_plt = [grad.detach().cpu().numpy() for grad in grad_plt]
    ax[0].semilogy(x_grad, grad_plt, linewidth=4, label=label)

    grad_list = sorted(grad_var[label].items())
    x_grad, grad_plt = zip(*grad_list)
    grad_plt = [grad.detach().cpu().numpy() for grad in grad_plt]
    ax[1].semilogy(x_grad, grad_plt, linewidth=4, label=label)

ax[0].legend(fontsize=14)
ax[0].set_title("Bias of Inference Network Gradient", fontsize=14)
ax[0].set_xlabel("Epoch", fontsize=14)
ax[0].set_ylabel("Log Bias", fontsize=14)

ax[1].legend(fontsize=14)
ax[1].set_title("Variance of Inference Network Gradient", fontsize=14)
ax[1].set_xlabel("Epoch", fontsize=14)
ax[1].set_ylabel("Log Variance", fontsize=14)

plt.savefig("figures/grad_stats/vscgrad.png", bbox_inches='tight')
plt.close()