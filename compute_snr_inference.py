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
num_samples = 1
sample_method = "max"
base_directory = "results"
trial = 1
num_forward_passes = 1000
base_lambda = 20
file_suffix = f"_{num_samples}samp_v{trial}"

file_list = [
    #f"{base_directory}/FISTA_fnorm1e-4_v{trial}/",
    f"{base_directory}/gaussian_{num_samples}samp_v{trial}/",
    f"{base_directory}/laplacian_{num_samples}samp_v{trial}/",
    f"{base_directory}/concreteslab_{num_samples}samp_v{trial}/",
    f"{base_directory}/gaussian_thresh_{num_samples}samp_v{trial}/",
    f"{base_directory}/gaussian_learnthresh_{num_samples}samp_v{trial}/",
    f"{base_directory}/laplacian_thresh_{num_samples}samp_v{trial}/",
    f"{base_directory}/laplacian_learnthresh_{num_samples}samp_v{trial}/",
]

file_labels = [
    #"FISTA",
    "Gaussian",
    "Laplacian",
    "Concreteslab",
    "Gaussian Thresh",
    "Gaussian Thresh+Gamma",
    "Laplacian Thresh",
    "Laplacian Thresh+Gamma"
]

# %%
#with open(base_run + "config.json") as json_data:
with open(file_list[0] + 'config.json') as json_data:
    config_data = json.load(json_data)
logging.basicConfig(filename=f"figures/snr/inference_snr{file_suffix}.txt", 
                    filemode='w', level=logging.DEBUG)
train_args = SimpleNamespace(**config_data['train'])
gt_dictionary = np.load(file_list[0] + 'train_savefile.npz')['phi'][-1]

default_device = torch.device('cuda:0')
_, val_patches = load_whitened_images(train_args, gt_dictionary)
p_signal = np.var(val_patches.reshape(len(val_patches), -1), axis=-1).mean()

# %%

for idx, train_run in enumerate(file_list):
    logging.info(f"Method {file_labels[idx]}")

    with open(train_run + 'config.json') as json_data:
        config_data = json.load(json_data)
    train_args = SimpleNamespace(**config_data['train'])
    solver_args = SimpleNamespace(**config_data['solver'])

    if solver_args.solver == "FISTA":
        epoch_list = np.arange(0, train_args.epochs + 1, 20)
    else:
        epoch_list = [int(re.search(r'epoch([0-9].*).pt', f)[1]) for f in os.listdir(train_run) if re.search(r'epoch([0-9].*).pt', f)]
        epoch_list = [300]

    for epoch in epoch_list:
        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)
        
        encoder = VIEncoder(train_args.patch_size**2, train_args.dict_size, solver_args).to(default_device)
        encoder.load_state_dict(torch.load(train_run + f"encoderstate_epoch{epoch}.pt", map_location=default_device)['model_state'])
        encoder.ramp_hyperparams()
        if solver_args.prior_distribution == "concreteslab":
            encoder.temp = solver_args.temp_min
        sgd = torch.optim.SGD(encoder.parameters(), lr=1e0)

        model_size = len(torch.cat([param.data.reshape(-1).detach().cpu() for param in encoder.parameters()]))

        if epoch == 0:
            phi = np.random.randn(train_args.patch_size ** 2, train_args.dict_size)
            phi /= np.sqrt(np.sum(phi ** 2, axis=0))
            phi = torch.tensor(phi, device=default_device).float()
        else:
            phi = torch.tensor(np.load(train_run + 'train_savefile.npz')['phi'][epoch - 1], device=default_device).float()

        dataset_mean_snr = []
        dataset_median_snr = []        
        for j in range(val_patches.shape[0] // train_args.batch_size):
            # Load next batch of validation patches
            patches = val_patches[j * train_args.batch_size:(j + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T
            patches_cu = torch.tensor(patches.T).float().to(default_device)

            dict_grad_list = np.zeros((num_forward_passes, model_size))
            for k in range(num_forward_passes):
                encoder.solver_args.sample_method = sample_method
                encoder.solver_args.num_samples = num_samples
                iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(patches_cu, phi.detach()) 

                sgd.zero_grad()
                iwae_loss.backward()
                model_grad = [param.grad.data.reshape(-1).detach().cpu() for param in encoder.parameters()]
                model_grad = torch.cat(model_grad)
                dict_grad_list[k] += model_grad.numpy() / (len(val_patches) // train_args.batch_size)

            grad_mean = np.mean(dict_grad_list, axis=0)
            grad_std = np.std(dict_grad_list, axis=0)
            grad_snr = np.abs(grad_mean / (grad_std + 1e-9))
            dataset_mean_snr.append(np.nanmean(grad_snr))
            dataset_median_snr.append(np.median(grad_snr))
    
        dataset_mean_snr = np.stack(dataset_mean_snr)
        dataset_median_snr = np.stack(dataset_median_snr)

        logging.info(f"Epoch {epoch}, (mean batch, mean param) snr: {np.nanmean(dataset_mean_snr):.3E}, " +
                     f"(med batch, mean param) snr: {np.median(dataset_mean_snr):.3E}, " + 
                     f"(mean batch, med param) snr: {np.nanmean(dataset_median_snr):.3E}, " +
                     f"(med batch, med param) snr: {np.median(dataset_median_snr):.3E}")
    logging.info("\n")

#np.savez_compressed(f"figures/grad_stats/dictgrad_{file_suffix}_save.npz",
#        grad_bias=grad_bias, grad_var=grad_var, 
#        residual_bias=residual_bias, residual_var=residual_var, 
#        image_bias=image_bias, image_var=image_var, 
#        file_list=file_list, file_labels=file_labels)
