"""
Train sparse dictionary model (Olshausen 1997) with whitened images used in the original paper. This script
applies a variational posterior to learn the sparse codes.

@Filename    train_vsc_dict.py
@Author      Kion
@Created     11/01/21
"""
import argparse
import time
import os
import logging
import json, codecs
from types import SimpleNamespace

import numpy as np
import torch

from compute_vsc_statistics import compute_statistics
from utils.dict_plotting import show_dict
from utils.solvers import FISTA, ADMM
from model.lista import VIEncoderLISTA, LISTA
from model.vi_encoder import VIEncoder
from model.scheduler import CycleScheduler
from utils.data_loader import load_whitened_images
from utils.util import *

# Load arguments for training via config file input to CLI #
parser = argparse.ArgumentParser(description='Variational Sparse Coding')
parser.add_argument('-c', '--config', type=str, required=True,
                    help='Path to config file for training.')
args = parser.parse_args()
with open(args.config) as json_data:
    config_data = json.load(json_data)
train_args = SimpleNamespace(**config_data['train'])
solver_args = SimpleNamespace(**config_data['solver'])

default_device = torch.device('cuda', train_args.device)
if not os.path.exists(train_args.save_path):
    os.makedirs(train_args.save_path)
    print("Created directory for figures at {}".format(train_args.save_path))
with open(train_args.save_path + '/config.json', 'wb') as f:
    json.dump(config_data, codecs.getwriter('utf-8')(f), ensure_ascii=False, indent=2)
logging.basicConfig(filename=os.path.join(train_args.save_path, 'training.log'), 
                    filemode='w', level=logging.DEBUG)

if __name__ == "__main__":
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    # INITIALIZE DICTIONARY #
    if train_args.fixed_dict:
        dictionary = np.load("data/ground_truth_dict.npy")
        step_size = 0.
    else:
        dictionary = np.random.randn(train_args.patch_size ** 2, train_args.dict_size)
        dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))
        step_size = train_args.lr

    train_patches, val_patches = load_whitened_images(train_args, dictionary)

    # INITIALIZE 
    if solver_args.solver == "VI":
        encoder = VIEncoder(train_args.patch_size**2, train_args.dict_size, solver_args).to(default_device)

        vi_opt = torch.optim.SGD(encoder.parameters(), lr=solver_args.vsc_lr, #weight_decay=1e-4,
                                    momentum=0.9, nesterov=True)
        vi_scheduler = CycleScheduler(vi_opt, solver_args.vsc_lr, 
                                        n_iter=(train_args.epochs * train_patches.shape[0]) // train_args.batch_size,
                                        momentum=None, warmup_proportion=0.05)

        # Create core-set for prior
        if solver_args.prior_method == "coreset":
            build_coreset(solver_args, encoder, train_patches, default_device)
        
        torch.save({'model_state': encoder.state_dict()}, train_args.save_path + "encoderstate_epoch0.pt")
    elif solver_args.solver == "FISTA" or solver_args.solver == "ADMM":
        lambda_warmup = 0.1

    # Initialize empty arrays for tracking learning data
    dictionary_saved = np.zeros((train_args.epochs, *dictionary.shape))
    dictionary_use = np.zeros((train_args.epochs, train_args.dict_size))
    lambda_list = np.zeros((train_args.epochs, train_args.dict_size))
    coeff_true = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
    coeff_est = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
    train_loss = np.zeros(train_args.epochs)
    val_true_recon, val_recon = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    val_true_l1, val_l1 = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    val_iwae_loss, val_kl_loss = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    train_time = np.zeros(train_args.epochs)

    # TRAIN MODEL #
    init_time = time.time()
    for j in range(train_args.epochs):
        epoch_loss = np.zeros(train_patches.shape[0] // train_args.batch_size)
        # Shuffle training data-set
        shuffler = np.random.permutation(len(train_patches))
        for i in range(train_patches.shape[0] // train_args.batch_size):
            patches = train_patches[shuffler][i * train_args.batch_size:(i + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T
            patch_idx = shuffler[i * train_args.batch_size:(i + 1) * train_args.batch_size]
            patches_cu = torch.tensor(patches.T).float().to(default_device)
            dict_cu = torch.tensor(dictionary, device=default_device).float()

            # Infer coefficients
            if solver_args.solver == "FISTA":
                b = FISTA(dictionary, patches, tau=solver_args.lambda_*lambda_warmup)
                b_select = np.array(b)
                b = torch.tensor(b, device=default_device).unsqueeze(dim=0).float()
                weight = torch.ones((len(b), 1), device=default_device)
                lambda_warmup += 1e-4
                if lambda_warmup >= 1.0:
                    lambda_warmup = 1.0
            elif solver_args.solver == "ADMM":
                b = ADMM(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "VI":
                iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(patches_cu, dict_cu, patch_idx)

                vi_opt.zero_grad()
                iwae_loss.backward()
                vi_opt.step()
                vi_scheduler.step()

                if solver_args.true_coeff and not train_args.fixed_dict:
                    b = FISTA(dictionary, patches, tau=solver_args.lambda_)
                else:
                    sample_idx = torch.distributions.categorical.Categorical(weight).sample().detach()
                    b_select = b_cu[torch.arange(len(b_cu)), sample_idx].detach().cpu().numpy().T
                    weight = weight.detach()
                    b = b_cu.permute(1, 2, 0).detach()

            # Take gradient step on dictionaries
            generated_patch = dict_cu @ b
            residual = patches_cu.T - generated_patch
            #select_penalty = np.sqrt(np.sum(dictionary ** 2, axis=0)) > 1.5
            step = ((residual[:, :, None] * b[:, None]) * weight.T[:, None, None]).sum(axis=(0, 3)) / train_args.batch_size
            step = step.detach().cpu().numpy() -  2*train_args.fnorm_reg*dictionary#*select_penalty
            dictionary += step_size * step
            
            # Normalize dictionaries. Required to prevent unbounded growth, Tikhonov regularisation also possible.
            if train_args.normalize:
                dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))

            # Calculate loss after gradient step
            epoch_loss[i] = 0.5 * np.sum((patches - dictionary @ b_select) ** 2) + solver_args.lambda_ * np.sum(np.abs(b_select))
            # Log which dictionary entries are used
            dict_use = np.count_nonzero(b_select, axis=1)
            dictionary_use[j] += dict_use / ((train_patches.shape[0] // train_args.batch_size))

            # Ramp up sigmoid for spike-slab
            if solver_args.prior_distribution == "concreteslab":
                encoder.temp *= 0.9995
                if encoder.temp <= solver_args.temp_min:
                    encoder.temp = solver_args.temp_min
            if solver_args.prior_method == "clf":
                encoder.clf_temp *= 0.9995
                if encoder.clf_temp <= solver_args.clf_temp_min:
                    encoder.clf_temp = solver_args.clf_temp_min
            if solver_args.prior_distribution == "concreteslab" or solver_args.prior_distribution == "laplacian":
                if ((train_patches.shape[0] // train_args.batch_size)*j + i) >= 1500:
                    encoder.warmup += 2e-4
                    if encoder.warmup >= 1.0:
                        encoder.warmup = 1.0

        # Test reconstructed or uncompressed dictionary on validation data-set
        epoch_true_recon = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_val_recon = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_true_l1 = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_val_l1 = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_iwae_loss = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_kl_loss = np.zeros(val_patches.shape[0] // train_args.batch_size)
        for i in range(val_patches.shape[0] // train_args.batch_size):
            # Load next batch of validation patches
            patches = val_patches[i * train_args.batch_size:(i + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T
            patch_idx = np.arange(i * train_args.batch_size, (i + 1) * train_args.batch_size)

            # Infer coefficients
            if solver_args.solver == "FISTA":
                b_hat = FISTA(dictionary, patches, tau=solver_args.lambda_)
                b_true = np.array(b_hat)
                iwae_loss, kl_loss = 0., np.array(0.)
            elif solver_args.solver == "ADMM":
                b_hat = ADMM(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "VI":
                with torch.no_grad():
                    patches_cu = torch.tensor(patches.T).float().to(default_device)
                    dict_cu = torch.tensor(dictionary, device=default_device).float()
                    iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(patches_cu, dict_cu, patch_idx)
                    sample_idx = torch.distributions.categorical.Categorical(weight).sample().detach()
                    b_select = b_cu[torch.arange(len(b_cu)), sample_idx]
                    b_hat = b_select.detach().cpu().numpy().T
                    b_true = FISTA(dictionary, patches, tau=solver_args.lambda_)

            # Compute and save loss
            epoch_true_recon[i] = 0.5 * np.sum((patches - dictionary @ b_true) ** 2)
            epoch_val_recon[i] = 0.5 * np.sum((patches - dictionary @ b_hat) ** 2)
            epoch_true_l1[i] = np.sum(np.abs(b_true))
            epoch_val_l1[i] = np.sum(np.abs(b_hat))
            epoch_iwae_loss[i], epoch_kl_loss[i]  = iwae_loss, kl_loss.mean()

        # Decay step-size
        step_size = step_size * train_args.lr_decay

        # Save and print data from epoch
        train_time[j] = time.time() - init_time
        epoch_time = train_time[0] if j == 0 else train_time[j] - train_time[j - 1]
        train_loss[j] = np.sum(epoch_loss) / len(train_patches)
        val_recon[j], val_l1[j] = np.sum(epoch_val_recon) / len(val_patches), np.sum(epoch_val_l1) / len(val_patches)
        val_true_recon[j], val_true_l1[j] = np.sum(epoch_true_recon) / len(val_patches), np.sum(epoch_true_l1) / len(val_patches)
        val_iwae_loss[j], val_kl_loss[j]  = np.mean(epoch_iwae_loss), np.mean(epoch_kl_loss)
        coeff_est[j], coeff_true[j] = b_hat.T, b_true.T
        if solver_args.threshold and solver_args.solver == "VI":
            lambda_list[j] = encoder.lambda_.data.mean(dim=(0, 1)).cpu().numpy()
        else:
            lambda_list[j] = np.ones(train_args.dict_size) * -1
        dictionary_saved[j] = dictionary

        if solver_args.debug:
            print_debug(train_args, b_true.T, b_hat.T)
            logging.info("Mean lambda value: {:.3E}".format(lambda_list[j].mean()))
            logging.info("Mean dict norm: {}".format(np.sqrt(np.sum(dictionary ** 2, axis=0)).mean()))
            logging.info("Est IWAE loss: {:.3E}".format(val_iwae_loss[j]))
            logging.info("Est KL loss: {:.3E}".format(val_kl_loss[j]))
            logging.info("Est total loss: {:.3E}".format(val_recon[j] + solver_args.lambda_ * val_l1[j]))
            logging.info("FISTA total loss: {:.3E}".format(val_true_recon[j] + solver_args.lambda_ * val_true_l1[j]))

        if j < 10 or (j + 1) % train_args.save_freq == 0 or (j + 1) == train_args.epochs:
            show_dict(dictionary, train_args.save_path + f"dictionary_epoch{j+1}.png")
            np.savez_compressed(train_args.save_path + f"train_savefile.npz",
                    phi=dictionary_saved, lambda_list=lambda_list, time=train_time,
                    train=train_loss, val_true_recon=val_true_recon, val_recon=val_recon, 
                    val_l1=val_l1, val_true_l1=val_true_l1, val_iwae_loss=val_iwae_loss,
                    val_kl_loss=val_kl_loss, coeff_est=coeff_est, coeff_true=coeff_true,
                    dictionary_use=dictionary_use)
            if solver_args.solver == "VI":
                torch.save({
                            'model_state': encoder.state_dict()
                            }, train_args.save_path + f"encoderstate_epoch{j+1}.pt")
        logging.info("Epoch {} of {}, Avg Train Loss = {:.4f}, Avg Val Loss = {:.4f}, Time = {:.0f} secs".format(j + 1,
                                                                                                          train_args.epochs,
                                                                                                          train_loss[j],
                                                                                                          val_recon[j] + solver_args.lambda_ * val_l1[j],
                                                                                                          epoch_time))
        logging.info("\n")

    if train_args.compute_stats:
        compute_statistics(train_args.save_path, train_args, solver_args)
