# %%
import json
import logging
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
distribution = "concreteslab"
run_name = "Concrete Slab 1 Samp"

num_samples = 1
sample_method = "max"
file_suffix = f"{distribution}_{num_samples}samp"
base_lambda = 8.0


# Coadapt baseline
file_list = [
    f"comp256_v4/{file_suffix}_v1/",
    f"comp256_v4/{file_suffix}_v2/",
    f"comp256_v4/{file_suffix}_v3/",

]

file_labels = [
    #"FISTA",
    run_name + " v1",
    run_name + " v2",
    run_name + " v3",
]

# %%
#with open(base_run + "config.json") as json_data:
with open(file_list[0] + 'config.json') as json_data:
    config_data = json.load(json_data)
logging.basicConfig(filename=f"figures/grad_stats/dictgrad_{file_suffix}.txt", 
                    filemode='w', level=logging.DEBUG)
train_args = SimpleNamespace(**config_data['train'])
gt_dictionary = np.load(file_list[0] + 'train_savefile.npz')['phi'][-1]

default_device = torch.device('cuda')
_, val_patches = load_whitened_images(train_args, gt_dictionary)
p_signal = np.var(val_patches.reshape(len(val_patches), -1), axis=-1).mean()

# %%
grad_bias = {}
grad_var = {}

residual_bias = {}
residual_var = {}

image_bias = {}
image_var = {}

for idx, train_run in enumerate(file_list):
    logging.info(f"Method {file_labels[idx]}")
    grad_bias[file_labels[idx]] = {}
    grad_var[file_labels[idx]] = {}
    residual_bias[file_labels[idx]] = {}
    residual_var[file_labels[idx]] = {}
    image_bias[file_labels[idx]] = {}
    image_var[file_labels[idx]] = {}

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
            image_bias[file_labels[idx]][epoch] = 0
            image_var[file_labels[idx]][epoch] = 0

            np.random.seed(train_args.seed)
            torch.manual_seed(train_args.seed)
            
            if solver_args.solver != "FISTA":
                encoder = VIEncoder(train_args.patch_size**2, train_args.dict_size, solver_args).to(default_device)
                encoder.load_state_dict(torch.load(train_run + f"encoderstate_epoch{epoch}.pt", map_location=default_device)['model_state'])
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
                    encoder.solver_args.sample_method = "max"
                    encoder.solver_args.num_samples = 500
                    iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(patches_cu, phi.detach()) 

                sample_idx = torch.distributions.categorical.Categorical(weight).sample().detach()
                b_cu = b_cu.permute(1, 0, 2).detach()
                true_residual = (patches_cu - b_cu @ phi.T)
                true_grad = ((true_residual * b_cu) * weight.T[..., None]).sum(dim=0) / (-0.5 * train_args.dict_size)
                #true_grad = b_cu.detach()[..., None] * true_residual[:, None] / (-0.5 *  256)
                true_grad = true_grad.detach().cpu()
                true_residual = true_residual[sample_idx, torch.arange(true_residual.shape[1])].detach().cpu()

                batch_var = []
                batch_residual = []
                image_est = []
                for k in range(300):
                    if solver_args.solver == "FISTA":
                        b = FISTA(phi.detach().cpu().numpy(), patches, tau=base_lambda)
                        b_cu = torch.tensor(b, device=default_device).float().T
                    elif solver_args.solver == "VI":
                        encoder.solver_args.sample_method = sample_method
                        encoder.solver_args.num_samples = num_samples
                        iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(patches_cu, phi.detach()) 

                    sample_idx = torch.distributions.categorical.Categorical(weight).sample().detach()
                    b_cu = b_cu.permute(1, 0, 2).detach()
                    x_hat = b_cu @ phi.T
                    residual = patches_cu - x_hat
                    model_grad = ((residual * b_cu) * weight.T[..., None]).sum(dim=0) / (-0.5 * train_args.dict_size)
                    model_grad = model_grad.detach().cpu()
                    x_hat = x_hat[sample_idx, torch.arange(x_hat.shape[1])]
                    residual = residual[sample_idx, torch.arange(residual.shape[1])]

                    batch_var.append(model_grad)
                    batch_residual.append(residual.detach().cpu())
                    image_est.append(x_hat.detach().cpu())

                batch_var = torch.stack(batch_var)
                batch_residual = torch.stack(batch_residual)
                image_est = torch.stack(image_est)

                grad_bias[file_labels[idx]][epoch] += torch.linalg.norm(true_grad - torch.mean(batch_var, dim=0)).mean()  / (val_patches.shape[0] // train_args.batch_size)
                grad_var[file_labels[idx]][epoch] += torch.var(batch_var, dim=0).sum() / (val_patches.shape[0] // train_args.batch_size)
                residual_bias[file_labels[idx]][epoch] += torch.linalg.norm(true_residual - torch.mean(batch_residual, dim=0)).mean() / (val_patches.shape[0] // train_args.batch_size)
                residual_var[file_labels[idx]][epoch] += torch.var(batch_residual, dim=0).sum() / (val_patches.shape[0] // train_args.batch_size)
                image_bias[file_labels[idx]][epoch] += torch.linalg.norm(patches_cu.detach().cpu() - torch.mean(image_est, dim=0), axis=1).mean() / (val_patches.shape[0] // train_args.batch_size)
                image_var[file_labels[idx]][epoch] += torch.var(image_est, dim=0).sum() / (val_patches.shape[0] // train_args.batch_size)

            logging.info(f"Epoch {epoch}, grad bias: {grad_bias[file_labels[idx]][epoch]:.3E}, grad var: {grad_var[file_labels[idx]][epoch]:.3E}, " + \
                         f"residual bias: {residual_bias[file_labels[idx]][epoch]:.3E}, residual var: {residual_var[file_labels[idx]][epoch]:.3E}, " + \
                         f"image bias: {image_bias[file_labels[idx]][epoch]:.3E}, image var: {image_var[file_labels[idx]][epoch]:.3E}")
    logging.info("\n")

np.savez_compressed(f"figures/grad_stats/dictgrad_{file_suffix}_save.npz",
        grad_bias=grad_bias, grad_var=grad_var, 
        residual_bias=residual_bias, residual_var=residual_var, 
        image_bias=image_bias, image_var=image_var, 
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

plt.savefig(f"figures/grad_stats/dictgrad_{file_suffix}.png", bbox_inches='tight')
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

plt.savefig(f"figures/grad_stats/dictresidual_{file_suffix}.png", bbox_inches='tight')
plt.close()

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

for idx, label in enumerate(file_labels):
    grad_list = sorted(image_bias[label].items())
    x_grad, grad_plt = zip(*grad_list)
    grad_plt = [grad.detach().cpu().numpy() for grad in grad_plt]
    ax[0].semilogy(x_grad, grad_plt, linewidth=4, label=label)

    grad_list = sorted(image_var[label].items())
    x_grad, grad_plt = zip(*grad_list)
    grad_plt = [grad.detach().cpu().numpy() for grad in grad_plt]
    ax[1].semilogy(x_grad, grad_plt, linewidth=4, label=label)

ax[0].legend(fontsize=14)
ax[0].set_title("Bias of Patch Estimates", fontsize=14)
ax[0].set_xlabel("Epoch", fontsize=14)
ax[0].set_ylabel("Log Bias", fontsize=14)

ax[1].legend(fontsize=14)
ax[1].set_title("Variance of Patch Estimates", fontsize=14)
ax[1].set_xlabel("Epoch", fontsize=14)
ax[1].set_ylabel("Log Variance", fontsize=14)

plt.savefig(f"figures/grad_stats/image_{file_suffix}.png", bbox_inches='tight')
plt.close()