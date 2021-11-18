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

import numpy as np
from pyitlib import discrete_random_variable as drv
from matplotlib import pyplot as plt

from utils.solvers import FISTA
from model.lista import VIEncoderLISTA, LISTA
from model.vi_encoder import VIEncoder

def compute_statistics(run_path, im_count, train_args, solver_args):
    phi = np.load("data/ground_truth_dict.npy")
    data_file = f"data/imagepatches_16.np"
    default_device = torch.device('cuda', train_args.device)
    assert os.path.exists(data_file), "Processed image patches needed in data folder."
    with open(data_file, 'rb') as f:
        data_patches = np.load(f)
    logging.basicConfig(filename=os.path.join(run_path, 'statistics.log'), 
                        filemode='w', level=logging.DEBUG)
    train_idx = np.linspace(1, data_patches.shape[0] - 1, im_count, dtype=int)
    random_images = data_patches[train_idx, :, :].reshape(-1, 16**2)
    random_images = random_images / np.linalg.norm(random_images, axis=1)[:, None]
    train_patches = np.array(random_images)

    load_list = [int(re.search(r'epoch([0-9].*).pt', f)[1]) for f in os.listdir(run_path) if re.search(r'epoch([0-9].*).pt', f)]
    load_list = np.sort(load_list)

    multi_info = np.zeros(len(load_list))
    posterior_collapse = np.zeros(len(load_list))
    coeff_collapse = np.zeros(len(load_list))
    code_list = np.zeros((len(load_list), train_patches.shape[0], phi.shape[1]))

    for idx, method in enumerate(load_list):
        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)

        kl_collapse_count = np.zeros(phi.shape[1])
        coeff_collapse_count = np.zeros(phi.shape[1])
        for i in range(train_patches.shape[0] // train_args.batch_size):
            patches = train_patches[i * train_args.batch_size:(i + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T

            with torch.no_grad():
                encoder = VIEncoder(train_args.patch_size, train_args.dict_size, solver_args).to(default_device)
                encoder.load_state_dict(torch.load(run_path  + f"/encoderstate_epoch{method}.pt")['model_state'])
                
                patches_cu = torch.tensor(patches.T).float().to(default_device)
                dict_cu = torch.tensor(phi, device=default_device).float()

                iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu)
                code_est = b_cu.detach().cpu().numpy()

                if solver_args.prior == "laplacian":
                    feat = encoder.enc(patches_cu)
                    logscale, mu = encoder.shift(feat), encoder.scale(feat)
                    scale = torch.exp(logscale)
                    kl = (-1 - logscale + mu.abs() + scale*(-mu.abs() / scale).exp())
                elif solver_args.prior == "gaussian":
                    feat = encoder.enc(patches_cu)
                    logvar, mu = encoder.shift(feat), encoder.scale(feat)        
                    kl = - 0.5 * (1 + logvar - (mu ** 2) - logvar.exp())
                elif solver_args.prior == "vampprior":
                    feat = encoder.enc(patches_cu)
                    logvar, mu = encoder.shift(feat), encoder.scale(feat)
                    
                    pseudo_feat = encoder.enc(encoder.pseudo_inputs)
                    pseudo_logvar, pseudo_mu = encoder.shift(pseudo_feat).unsqueeze(0), encoder.scale(pseudo_feat).unsqueeze(0)               

                    scale = torch.exp(0.5*logvar)
                    eps = torch.randn_like(scale)
                    z = mu + eps*scale 

                    log_p_z = -0.5 * (pseudo_logvar + torch.pow(z.unsqueeze(1) - pseudo_mu, 2 ) / torch.exp(pseudo_logvar))
                    log_p_z = torch.logsumexp(log_p_z - np.log(solver_args.num_pseudo_inputs), dim=-2)
                    log_q_z = -0.5 * (logvar + torch.pow(z - mu, 2 ) / torch.exp(logvar))
                    kl = log_q_z - log_p_z

            for k in range(phi.shape[1]):
                kl_collapse_count[k] += (kl[:, k] <= 1e-2).sum() / train_patches.shape[0]
                coeff_collapse_count[k] += (np.abs(code_est[:, k]) <= 1e-2).sum() / train_patches.shape[0]
            code_list[idx, i*train_args.batch_size:(i+1)*train_args.batch_size] = code_est

        posterior_collapse[idx] = 100. * (kl_collapse_count >= 0.95).sum() / phi.shape[1]
        coeff_collapse[idx] = 100. * (coeff_collapse_count >= 0.95).sum() / phi.shape[1]
        multi_info[idx] = drv.information_multi(code_list[idx].T)
        logging.info(f"Epoch {method}, multi-information: {multi_info[idx]:.3E}, % posterior collapse: {posterior_collapse[idx]:.2f}%, % coeff collapse: {coeff_collapse[idx]:.2f}%")
    np.savez_compressed(run_path + "/encoder_statistics.npz",
        code_list=code_list, posterior_collapse=posterior_collapse, 
        coeff_collapse=coeff_collapse, multi_info=multi_info, load_list=load_list)

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