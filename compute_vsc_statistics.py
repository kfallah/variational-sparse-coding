"""
Train sparse dictionary model (Olshausen 1997) with whitened images used in the original paper. This script
applies a variational posterior to learn the sparse codes.

@Filename    train_sparse_dict
@Author      Kion
@Created     5/29/20
"""
import json
import re
import os
import argparse
import logging
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import numpy as np
from pyitlib import discrete_random_variable as drv
from matplotlib import pyplot as plt

from utils.solvers import FISTA
from utils.dict_plotting import show_dict
from model.lista import VIEncoderLISTA, LISTA
from model.vi_encoder import VIEncoder

def compute_statistics(run_path, im_count, train_args, solver_args):
    solver_args.iwae = True
    solver_args.num_samples = 20

    phi = np.load("data/mbedpatches_11.npz")['phi']
    data_file = f"data/imagepatches_11.np"
    default_device = torch.device('cuda', train_args.device)
    assert os.path.exists(data_file), "Processed image patches needed in data folder."
    with open(data_file, 'rb') as f:
        data_patches = np.load(f)
    logging.basicConfig(filename=os.path.join(run_path, 'statistics.log'), 
                        filemode='w', level=logging.DEBUG)
    train_idx = np.linspace(1, data_patches.shape[0] - 1, im_count, dtype=int)
    train_patches = data_patches[train_idx, :, :].reshape(-1, 11**2)
    train_mean, train_std = np.mean(train_patches, axis=0), np.std(train_patches, axis=0)
    train_patches = (train_patches - train_mean) / train_std

    load_list = [int(re.search(r'epoch([0-9].*).pt', f)[1]) for f in os.listdir(run_path) if re.search(r'epoch([0-9].*).pt', f)]
    load_list = np.sort(load_list)

    multi_info = np.zeros(len(load_list))
    posterior_collapse = np.zeros(len(load_list))
    coeff_collapse = np.zeros(len(load_list))
    iwae_likelihood = np.zeros(len(load_list))
    code_list = np.zeros((len(load_list), train_patches.shape[0], phi.shape[1]))
    recovered_dict = np.zeros((len(load_list), *phi.shape))

    for idx, method in enumerate(load_list):
        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)

        encoder = VIEncoder(train_args.patch_size, train_args.dict_size, solver_args).to(default_device)
        encoder.load_state_dict(torch.load(run_path  + f"/encoderstate_epoch{method}.pt")['model_state'])

        kl_collapse_count = np.zeros(phi.shape[1])
        coeff_collapse_count = np.zeros(phi.shape[1])
        for i in range(train_patches.shape[0] // train_args.batch_size):
            patches = train_patches[i * train_args.batch_size:(i + 1) * train_args.batch_size].T

            if solver_args.solver == 'FISTA':
                code_est = FISTA(phi, patches, tau=solver_args.lambda_).T
                kl_loss = 1e99 * np.ones_like(code_est)
            else:
                with torch.no_grad():
                    patches_cu = torch.tensor(patches.T).float().to(default_device)
                    dict_cu = torch.tensor(phi, device=default_device).float()

                    iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu)
                    code_est = b_cu.detach().cpu().numpy()

            for k in range(phi.shape[1]):
                kl_collapse_count[k] += (kl_loss[:, k] <= 1e-2).sum() / train_patches.shape[0]
                coeff_collapse_count[k] += (np.abs(code_est[:, k]) <= 1e-2).sum() / train_patches.shape[0]
            
            code_list[idx, i*train_args.batch_size:(i+1)*train_args.batch_size] = code_est
            iwae_likelihood[idx] += iwae_loss.detach().cpu()

        C_sr = (train_patches.T @ code_list[idx]) / len(train_patches)
        C_rr = (code_list[idx].T @ code_list[idx]) / len(train_patches)

        posterior_collapse[idx] = 100. * (kl_collapse_count >= 0.95).sum() / phi.shape[1]
        coeff_collapse[idx] = 100. * (coeff_collapse_count >= 0.95).sum() / phi.shape[1]
        multi_info[idx] = drv.information_multi(code_list[idx].T)
        iwae_likelihood[idx] /= (train_patches.shape[0] // train_args.batch_size)
        recovered_dict[idx] = C_sr @ np.linalg.pinv(C_rr)
        show_dict(C_sr @ np.linalg.pinv(C_rr), train_args.save_path + f"recovered_dict{method}.png")

        logging.info(f"Epoch {method}, multi-information: {multi_info[idx]:.3E}, % posterior collapse: {posterior_collapse[idx]:.2f}%," +\
                     f" % coeff collapse: {coeff_collapse[idx]:.2f}%, iwae likelihood: {iwae_likelihood[idx]:.3E}")
    np.savez_compressed(run_path + "/encoder_statistics.npz",
        code_list=code_list, posterior_collapse=posterior_collapse, 
        coeff_collapse=coeff_collapse, multi_info=multi_info, 
        iwae_likelihood=iwae_likelihood, load_list=load_list,
        recovered_dict=recovered_dict)

if __name__ == "__main__":
    # Load arguments for training via config file input to CLI #
    parser = argparse.ArgumentParser(description='Compute VSC Statistics')
    parser.add_argument('-r', '--run', type=str, required=True,
                        help='Path to run file to compute statistics for.')
    parser.add_argument('-n', '--image_count', type=int, default=40000,
                        help='Number of samples to compute statistics from')
    args = parser.parse_args()
    with open(args.run + "/config.json") as json_data:
        config_data = json.load(json_data)
    train_args = SimpleNamespace(**config_data['train'])
    solver_args = SimpleNamespace(**config_data['solver'])
    
    compute_statistics(args.run, args.image_count, train_args, solver_args)