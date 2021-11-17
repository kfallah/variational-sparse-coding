"""
Train sparse dictionary model (Olshausen 1997) with whitened images used in the original paper. This script
applies a variational posterior to learn the sparse codes.

@Filename    train_sparse_dict
@Author      Kion
@Created     5/29/20
"""
import argparse
import time
import os
import logging
import json, codecs

import numpy as np
import scipy.io
from types import SimpleNamespace
from sklearn.feature_extraction.image import extract_patches_2d

import torch
import torch.nn.functional as F

from utils.dict_plotting import show_dict
from utils.solvers import FISTA, ADMM
from model.lista import VIEncoderLISTA, LISTA
from model.vi_encoder import VIEncoder
from model.scheduler import CycleScheduler
from utils.util import *

# Load arguments for training via config file input to CLI #
parser = argparse.ArgumentParser(description='TransOp Backbone Training')
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

    # LOAD DATA #
    data_matlab = scipy.io.loadmat('./data/whitened_images.mat')
    images = np.ascontiguousarray(data_matlab['IMAGES'])

    # Extract patches using SciKit-Learn. Out of 10 images, 8 are used for training and 2 are used for validation.
    data_file = f"data/imagepatches_{train_args.patch_size}.np"
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data_patches = np.load(f)
            val_patches = np.load(f)
    else:
        data_patches = np.moveaxis(extract_patches_2d(images[:, :, :-2], (train_args.patch_size, train_args.patch_size)), -1, 1). \
            reshape(-1, train_args.patch_size, train_args.patch_size)
        val_patches = extract_patches_2d(images[:, :, -2], (train_args.patch_size, train_args.patch_size))
        with open(data_file, 'wb') as f:
            np.save(f, data_patches)
            np.save(f, val_patches)

    # INITIALIZE DICTIONARY #
    if train_args.fixed_dict:
        dictionary = np.load("data/ground_truth_dict.npy")
        step_size = 0.
    else:
        dictionary = np.random.randn(train_args.patch_size ** 2, train_args.dict_size)
        dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))
        step_size = train_args.lr

    # LOAD DATASET #
    if not train_args.synthetic_data:
        # Designate patches to use for training & validation. This step will also normalize the data.
        val_patches = val_patches[np.linspace(1, val_patches.shape[0] - 1, train_args.val_samples, dtype=int), :, :]
        val_patches = val_patches / np.linalg.norm(val_patches.reshape(-1, train_args.patch_size ** 2), axis=1)[:, None, None]
        logging.info("Shape of validation dataset: {}".format(val_patches.shape))

        train_idx = np.linspace(1, data_patches.shape[0] - 1, train_args.train_samples, dtype=int)
        train_patches = data_patches[train_idx, :, :]
        train_patches = train_patches / np.linalg.norm(train_patches.reshape(-1, train_args.patch_size ** 2), axis=1)[:, None, None]
        logging.info("Shape of training dataset: {}".format(train_patches.shape))
    else:
        # Generate synthetic examples from ground truth dictionary
        assert train_args.fixed_dict, "Cannot train with synthetic examples when not using a fixed dictionary."
        val_codes = np.zeros((train_args.val_samples, dictionary.shape[1]))
        val_support = np.random.randint(0, high=dictionary.shape[1], size=(train_args.val_samples, train_args.synthetic_support))
        for i in range(train_args.val_samples):
            val_codes[i, val_support[i]] = np.random.randn(train_args.synthetic_support)
        val_patches = val_codes @ dictionary.T

        train_codes = np.zeros((train_args.train_samples, dictionary.shape[1]))
        train_support = np.random.randint(0, high=dictionary.shape[1], size=(train_args.train_samples, train_args.synthetic_support))
        for i in range(train_args.train_samples):
            train_codes[i, train_support[i]] = np.random.randn(train_args.synthetic_support)
        train_patches = train_codes @ dictionary.T

    # INITIALIZE 
    if solver_args.solver == "VI":
        encoder = VIEncoder(train_args.patch_size, train_args.dict_size, solver_args).to(default_device)
    elif solver_args.solver  == "VILISTA":
        encoder = VIEncoderLISTA(train_args.patch_size, train_args.dict_size, 
                                 solver_args.lambda_, solver_args).to(default_device)
    elif solver_args.solver  == "LISTA":
        encoder = LISTA(train_args.patch_size, train_args.dict_size, 
                        solver_args.lambda_).double().to(default_device)

    if solver_args.solver != "FISTA":
        vi_opt = torch.optim.SGD(encoder.parameters(), lr=1e-3, #weight_decay=1e-4,
                                    momentum=0.9, nesterov=True)
        vi_scheduler = CycleScheduler(vi_opt, 1e-3, 
                                        n_iter=(train_args.epochs * train_patches.shape[0]) // train_args.batch_size,
                                        momentum=None, warmup_proportion=0.05)

    # Initialize empty arrays for tracking learning data
    dictionary_saved = np.zeros((train_args.epochs, *dictionary.shape))
    coeff_true = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
    coeff_est = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
    train_loss = np.zeros(train_args.epochs)
    val_true_recon = np.zeros(train_args.epochs)
    val_recon = np.zeros(train_args.epochs)
    val_true_l1 = np.zeros(train_args.epochs)
    val_l1 = np.zeros(train_args.epochs)
    val_kl_loss = np.zeros(train_args.epochs)
    train_time = np.zeros(train_args.epochs)

    # TRAIN MODEL #
    init_time = time.time()
    for j in range(train_args.epochs):
        epoch_loss = np.zeros(train_patches.shape[0] // train_args.batch_size)
        # Shuffle training data-set
        np.random.shuffle(train_patches)
        for i in range(train_patches.shape[0] // train_args.batch_size):
            patches = train_patches[i * train_args.batch_size:(i + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T

            # Infer coefficients
            if solver_args.solver == "FISTA":
                b = FISTA(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "ADMM":
                b = ADMM(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "VI" or solver_args.solver == "VILISTA":
                if i == 0 and solver_args.gradient_variance:
                    var_samples = 50
                else:
                    var_samples = 1
                grad_list = []
                for s in range(var_samples):
                    patches_cu = torch.tensor(patches.T).float().to(default_device)
                    dict_cu = torch.tensor(dictionary, device=default_device).float()
                    iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu)

                    vi_opt.zero_grad()
                    iwae_loss.backward()

                    model_grad = [param.grad.data.reshape(-1) for param in encoder.parameters()]
                    model_grad = torch.cat(model_grad)
                    grad_list.append(model_grad.detach().cpu().numpy())

                grad_list = np.stack(grad_list)
                if i == 0 and solver_args.gradient_variance:
                    logging.info(f"GRAD VARIANCE: {np.var(grad_list, axis=0).mean():.4E}")
                vi_opt.step()
                vi_scheduler.step()
                if solver_args.true_coeff and not train_args.fixed_dict:
                    b = FISTA(dictionary, patches, tau=solver_args.lambda_)
                else:
                    b = b_cu.detach().cpu().numpy().T
            elif solver_args.solver == "LISTA":
                b = FISTA(dictionary, patches, tau=solver_args.lambda_)
                patches_cu = torch.tensor(patches.T).to(default_device)
                kl_loss, b_cu = encoder(patches_cu)

                vi_opt.zero_grad()
                recon_loss = F.mse_loss(b_cu, torch.tensor(b, device=b_cu.device).T)
                recon_loss.backward()
                vi_opt.step()
                vi_scheduler.step()

            # Take gradient step on dictionaries
            generated_patch = dictionary @ b
            residual = patches - generated_patch
            step = residual @ b.T
            dictionary += step_size * step
            # Normalize dictionaries. Required to prevent unbounded growth, Tikhonov regularisation also possible.
            dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))

            # Calculate loss after gradient step
            epoch_loss[i] = 0.5 * np.sum((patches - dictionary @ b) ** 2) + solver_args.lambda_ * np.sum(np.abs(b))

        # Test reconstructed or uncompressed dictionary on validation data-set
        epoch_true_recon = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_val_recon = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_true_l1 = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_val_l1 = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_kl_loss = np.zeros(val_patches.shape[0] // train_args.batch_size)
        for i in range(val_patches.shape[0] // train_args.batch_size):
            # Load next batch of validation patches
            patches = val_patches[i * train_args.batch_size:(i + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T

            # Infer coefficients
            if solver_args.solver == "FISTA":
                b_hat = FISTA(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "ADMM":
                b_hat = ADMM(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "VI" or solver_args.solver == "VILISTA":
                with torch.no_grad():
                    patches_cu = torch.tensor(patches.T).float().to(default_device)
                    dict_cu = torch.tensor(dictionary, device=default_device).float()
                    iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu)
                    b_hat = b_cu.detach().cpu().numpy().T
                    b_true = FISTA(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.lambda_ == "LISTA":
                with torch.no_grad():
                    patches_cu = torch.tensor(patches.T).to(default_device)
                    kl_loss, b_cu = encoder(patches_cu)
                    b_hat = b_cu.detach().cpu().numpy().T
                    b_true = FISTA(dictionary, patches, tau=solver_args.lambda_)

            # Compute and save loss
            epoch_true_recon[i] = 0.5 * np.sum((patches - dictionary @ b_true) ** 2)
            epoch_val_recon[i] = 0.5 * np.sum((patches - dictionary @ b_hat) ** 2)
            epoch_true_l1[i] = solver_args.lambda_ * np.sum(np.abs(b_true))
            epoch_val_l1[i] = solver_args.lambda_ * np.sum(np.abs(b_hat))
            epoch_kl_loss[i] = kl_loss

        # Decay step-size
        step_size = step_size * train_args.lr_decay

        # Ramp up sigmoid for spike-slab
        if solver_args.prior == "spikeslab":
            encoder.c *= 1.02
            if encoder.c >= solver_args.c_max:
                encoder.c = solver_args.c_max
        elif solver_args.prior == "concreteslab":
            encoder.temp *= 0.9
            if encoder.temp <= solver_args.temp_min:
                encoder.temp = solver_args.temp_min

        # Save and print data from epoch
        train_time[j] = time.time() - init_time
        epoch_time = train_time[0] if j == 0 else train_time[j] - train_time[j - 1]
        train_loss[j] = np.mean(epoch_loss)
        val_true_recon[j] = np.mean(epoch_true_recon)
        val_recon[j] = np.mean(epoch_val_recon)
        val_true_l1[j] = np.mean(epoch_true_l1)
        val_l1[j] = np.mean(epoch_val_l1)
        val_kl_loss[j] = np.mean(epoch_kl_loss)
        coeff_est[j] = b_hat.T
        coeff_true[j] = b_true.T
        dictionary_saved[j] = dictionary

        if solver_args.debug:
            print_debug(train_args, b_true.T, b_hat.T)
            logging.info("Est KL loss: {:.3E}".format(val_kl_loss[j]))
            logging.info("Est total loss: {:.3E}".format(val_recon[j] + solver_args.lambda_ * val_l1[j]))
            logging.info("True total loss: {:.3E}".format(val_l1[j] + solver_args.lambda_ * val_true_l1[j]))

        if j % train_args.save_freq == 0 or j == train_args.epochs-1:
            show_dict(dictionary, train_args.save_path + f"dictionary_epoch{j}.png")
            np.savez_compressed(train_args.save_path + f"savefile_epoch{j}.npz",
                    phi=dictionary_saved, time=train_time, train=train_loss, 
                    val_true_recon=val_true_recon, val_recon=val_recon, 
                    val_l1=val_l1, val_true_l1=val_true_l1,
                    val_kl_loss=val_kl_loss, coeff_est=coeff_est, coeff_true=coeff_true)
            torch.save({
                        'model_state': encoder.state_dict()
                        }, train_args.save_path + f"encoderstate_epoch{j}.pt")
        logging.info("Epoch {} of {}, Avg Train Loss = {:.4f}, Avg Val Loss = {:.4f}, Time = {:.0f} secs".format(j + 1,
                                                                                                          train_args.epochs,
                                                                                                          train_loss[j],
                                                                                                          val_recon[j] + solver_args.lambda_ * val_l1[j],
                                                                                                          epoch_time))
        logging.info("\n")
