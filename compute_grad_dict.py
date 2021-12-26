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
    #"prior_comp/FISTA_fnorm1e-4/",
    #"prior_comp/concreteslab_gradvar/",
    #"prior_comp/concreteslab_gumbel/",
    "prior_comp/concreteslab_raogumbel/",
    "prior_comp/gaussian_nothresh_iwae/",
    "prior_comp/gaussian_gradvar/",
    "prior_comp/laplacian_fixed_iwae/",
    "prior_comp/laplacian_thresh_iwae/",
]

file_labels = [
    #"FISTA",
    #"CS GumbelSoftmax Estimator",
    #"CS StraightThrough Estimator",
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

default_device = torch.device('cuda:0')
_, val_patches = load_whitened_images(train_args, gt_dictionary)
p_signal = np.var(val_patches.reshape(len(val_patches), -1), axis=-1).mean()

# %%
grad_bias = {}
grad_var = {}

residual_bias = {}
residual_var = {}

for idx, train_run in enumerate(file_list):
    print(f"Method {file_labels[idx]}")
    grad_bias[file_labels[idx]] = {}
    grad_var[file_labels[idx]] = {}
    residual_bias[file_labels[idx]] = {}
    residual_var[file_labels[idx]] = {}

    with open(train_run + 'config.json') as json_data:
        config_data = json.load(json_data)
    train_args = SimpleNamespace(**config_data['train'])
    solver_args = SimpleNamespace(**config_data['solver'])

    if solver_args.solver == "FISTA":
        epoch_list = np.arange(0, train_args.epochs + 1, 20)
    else:
        epoch_list = [int(re.search(r'epoch([0-9].*).pt', f)[1]) for f in os.listdir(train_run) if re.search(r'epoch([0-9].*).pt', f)]
        epoch_list = np.sort(epoch_list)

    for epoch in epoch_list:
        with torch.no_grad():
            grad_bias[file_labels[idx]][epoch] = 0
            grad_var[file_labels[idx]][epoch] = 0
            residual_bias[file_labels[idx]][epoch] = 0
            residual_var[file_labels[idx]][epoch] = 0

            np.random.seed(train_args.seed)
            torch.manual_seed(train_args.seed)
            
            if solver_args.solver != "FISTA":
                encoder = VIEncoder(train_args.patch_size, train_args.dict_size, solver_args).to(default_device)
                encoder.load_state_dict(torch.load(train_run + f"encoderstate_epoch{epoch}.pt")['model_state'])
                encoder.ramp_hyperparams()

            if epoch == 0:
                phi = np.random.randn(train_args.patch_size ** 2, train_args.dict_size)
                phi /= np.sqrt(np.sum(phi ** 2, axis=0))
                phi = torch.tensor(phi, device=default_device).float()
            else:
                phi = torch.tensor(np.load(train_run + 'train_savefile.npz')['phi'][epoch - 1], device=default_device).float()

            for j in range(val_patches.shape[0] // train_args.batch_size):
                # Load next batch of validation patches
                patches = val_patches[j * train_args.batch_size:(j + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T
                patches_cu = torch.tensor(patches.T).float().to(default_device)

                if solver_args.solver == "FISTA":
                    b = FISTA(phi.detach().cpu().numpy(), patches, tau=base_lambda)
                    b_cu = torch.tensor(b, device=default_device).float().T
                elif solver_args.solver == "VI":
                    encoder.solver_args.iwae = True
                    encoder.solver_args.num_samples = 500
                    iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, phi.detach()) 
                true_residual = (patches_cu - b_cu.detach() @ phi.T)
                true_grad = b_cu.detach()[..., None] * true_residual[:, None] / (-0.5 *  121)
                true_grad = true_grad.detach().cpu()
                true_residual = true_residual.detach().cpu()

                batch_var = []
                batch_residual = []
                for k in range(300):
                    if solver_args.solver == "FISTA":
                        b = FISTA(phi.detach().cpu().numpy(), patches, tau=base_lambda)
                        b_cu = torch.tensor(b, device=default_device).float().T
                    elif solver_args.solver == "VI":
                        encoder.solver_args.iwae = False
                        encoder.solver_args.num_samples = 20
                        iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, phi.detach()) 

                    residual = (patches_cu - b_cu.detach() @ phi.T)
                    model_grad = (b_cu.detach()[..., None] * residual[:, None] / (-0.5 *  121)).detach().cpu()
                    batch_var.append(model_grad)
                    batch_residual.append(residual.detach().cpu())

                batch_var = torch.stack(batch_var)
                batch_residual = torch.stack(batch_residual)

                grad_bias[file_labels[idx]][epoch] += torch.linalg.norm(true_grad - torch.mean(batch_var, dim=0)) / (val_patches.shape[0] // train_args.batch_size)
                grad_var[file_labels[idx]][epoch] += torch.var(batch_var, dim=0).mean() / (val_patches.shape[0] // train_args.batch_size)
                residual_bias[file_labels[idx]][epoch] += torch.linalg.norm(true_residual - torch.mean(batch_residual, dim=0)) / (val_patches.shape[0] // train_args.batch_size)
                residual_var[file_labels[idx]][epoch] += torch.var(batch_residual, dim=0).mean() / (val_patches.shape[0] // train_args.batch_size)

            print(f"Epoch {epoch}, grad bias: {grad_bias[file_labels[idx]][epoch]:.3E}, grad var: {grad_var[file_labels[idx]][epoch]:.3E}, " + \
                  f"residual bias: {residual_bias[file_labels[idx]][epoch]:.3E}, residual var: {residual_var[file_labels[idx]][epoch]:.3E}")
    print()

np.savez_compressed("figures/grad_stats/dictgrad_20samp_save.npz",
        grad_bias=grad_bias, grad_var=grad_var, 
        residual_bias=grad_bias, residual_var=grad_var, 
        file_list=file_list, file_labels=file_labels)


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
ax[0].set_title("Bias of Dictionary Gradient", fontsize=14)
ax[0].set_xlabel("Epoch", fontsize=14)
ax[0].set_ylabel("Log Bias", fontsize=14)

ax[1].legend(fontsize=14)
ax[1].set_title("Variance of Dictionary Gradient", fontsize=14)
ax[1].set_xlabel("Epoch", fontsize=14)
ax[1].set_ylabel("Log Variance", fontsize=14)

plt.savefig("figures/grad_stats/dictgrad_20samp.png", bbox_inches='tight')
plt.close()

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

for idx, label in enumerate(file_labels):
    grad_list = sorted(residual_bias[label].items())
    x_grad, grad_plt = zip(*grad_list)
    grad_plt = [grad.detach().cpu().numpy() for grad in grad_plt]
    ax[0].semilogy(x_grad, grad_plt, linewidth=4, label=label)

    grad_list = sorted(residual_var[label].items())
    x_grad, grad_plt = zip(*grad_list)
    grad_plt = [grad.detach().cpu().numpy() for grad in grad_plt]
    ax[1].semilogy(x_grad, grad_plt, linewidth=4, label=label)

ax[0].legend(fontsize=14)
ax[0].set_title("Bias of Residuals", fontsize=14)
ax[0].set_xlabel("Epoch", fontsize=14)
ax[0].set_ylabel("Log Bias", fontsize=14)

ax[1].legend(fontsize=14)
ax[1].set_title("Variance of Residuals", fontsize=14)
ax[1].set_xlabel("Epoch", fontsize=14)
ax[1].set_ylabel("Log Variance", fontsize=14)

plt.savefig("figures/grad_stats/dictresidual_20samp.png", bbox_inches='tight')
plt.close()