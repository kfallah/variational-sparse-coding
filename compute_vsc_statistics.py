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
from utils.data_loader import load_whitened_images
from utils.dict_plotting import show_dict
from model.lista import VIEncoderLISTA, LISTA
from model.vi_encoder import VIEncoder

def compute_statistics(run_path, train_args, solver_args):
    solver_args.iwae = False
    solver_args.num_samples = 1

    final_phi = np.load(train_args.save_path + 'train_savefile.npz')['phi'][train_args.epochs - 1]
    _, val_patches = load_whitened_images(train_args, final_phi)
    val_patches = val_patches.reshape(-1, train_args.patch_size**2)
    default_device = torch.device('cuda', train_args.device)
    
    if solver_args.solver != 'FISTA':
        load_list = [int(re.search(r'epoch([0-9].*).pt', f)[1]) for f in os.listdir(run_path) if re.search(r'epoch([0-9].*).pt', f)]
        load_list = np.sort(load_list)
    else:
        load_list = np.arange(0, 201, 20)

    multi_info = np.zeros(len(load_list))
    posterior_collapse = np.zeros(len(load_list))
    coeff_collapse = np.zeros(len(load_list))
    code_list = np.zeros((len(load_list), val_patches.shape[0], final_phi.shape[1]))
    recovered_dict = np.zeros((len(load_list), *final_phi.shape))

    for idx, method in enumerate(load_list):
        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)
        
        if solver_args.solver != 'FISTA':
            encoder = VIEncoder(train_args.patch_size, train_args.dict_size, solver_args).to(default_device)
            encoder.load_state_dict(torch.load(run_path  + f"/encoderstate_epoch{method}.pt")['model_state'])
            encoder.ramp_hyperparams()

        if method == 0:
            phi = np.random.randn(train_args.patch_size ** 2, train_args.dict_size)
            phi /= np.sqrt(np.sum(phi ** 2, axis=0))
        else:
            phi = np.load(train_args.save_path + 'train_savefile.npz')['phi'][method - 1]

        kl_collapse_count = np.zeros(final_phi.shape[1])
        coeff_collapse_count = np.zeros(final_phi.shape[1])
        for i in range(val_patches.shape[0] // train_args.batch_size):
            patches = val_patches[i * train_args.batch_size:(i + 1) * train_args.batch_size].T

            if solver_args.solver == 'FISTA':
                code_est = FISTA(phi, patches, tau=solver_args.lambda_).T
                kl_loss = np.zeros_like(code_est)
            else:
                with torch.no_grad():
                    patches_cu = torch.tensor(patches.T).float().to(default_device)
                    dict_cu = torch.tensor(phi, device=default_device).float()
                    iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu)
                    code_est = b_cu.detach().cpu().numpy()

            for k in range(phi.shape[1]):
                kl_collapse_count[k] += (kl_loss[:, k] <= 1e-2).sum() / val_patches.shape[0]
                coeff_collapse_count[k] += (np.abs(code_est[:, k]) <= 1e-2).sum() / val_patches.shape[0]
            
            code_list[idx, i*train_args.batch_size:(i+1)*train_args.batch_size] = code_est

        C_sr = (val_patches.T @ code_list[idx]) / len(val_patches)
        C_rr = (code_list[idx].T @ code_list[idx]) / len(val_patches)

        posterior_collapse[idx] = 100. * (kl_collapse_count >= 0.95).sum() / phi.shape[1]
        coeff_collapse[idx] = 100. * (coeff_collapse_count >= 0.95).sum() / phi.shape[1]

        bins = np.linspace(-2, 2, 20)
        bins = np.sort(np.append(bins, [-1e-50, 1e-50]))
        alphabet = np.tile(np.arange(len(bins)+1), (code_list[idx].shape[1], 1))   
        discrete_codes = np.digitize(code_list[idx].T, bins)
        multi_info[idx] = drv.information_multi(discrete_codes, Alphabet_X=alphabet)

        recovered_dict[idx] = C_sr @ np.linalg.pinv(C_rr)
        show_dict(C_sr @ np.linalg.pinv(C_rr), train_args.save_path + f"recovered_dict{method}.png")

        logging.info(f"Epoch {method}, multi-information: {multi_info[idx]:.3E}, % posterior collapse: {posterior_collapse[idx]:.2f}%," +\
                     f" % coeff collapse: {coeff_collapse[idx]:.2f}%")
    np.savez_compressed(run_path + "/encoder_statistics.npz",
        code_list=code_list, posterior_collapse=posterior_collapse, 
        coeff_collapse=coeff_collapse, multi_info=multi_info, 
        load_list=load_list, ecovered_dict=recovered_dict)

if __name__ == "__main__":
    # Load arguments for training via config file input to CLI #
    parser = argparse.ArgumentParser(description='Compute VSC Statistics')
    parser.add_argument('-r', '--run', type=str, required=True,
                        help='Path to run file to compute statistics for.')
    args = parser.parse_args()
    with open(args.run + "/config.json") as json_data:
        config_data = json.load(json_data)
    train_args = SimpleNamespace(**config_data['train'])
    solver_args = SimpleNamespace(**config_data['solver'])
    logging.basicConfig(filename=os.path.join(train_args.save_path, 'statistics.log'), 
                    filemode='w', level=logging.DEBUG)
    compute_statistics(args.run, train_args, solver_args)