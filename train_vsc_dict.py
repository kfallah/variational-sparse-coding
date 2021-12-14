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
from sklearn_extra.cluster import KMedoids

import torch
import torch.nn.functional as F

from compute_vsc_statistics import compute_statistics
from utils.dict_plotting import show_dict
from utils.solvers import FISTA, ADMM
from model.lista import VIEncoderLISTA, LISTA
from model.vi_encoder import VIEncoder
from model.scheduler import CycleScheduler
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

    # LOAD DATA #
    data_matlab = scipy.io.loadmat('./data/whitened_images.mat')
    images = np.ascontiguousarray(data_matlab['IMAGES'])

    # Extract patches using SciKit-Learn. Out of 10 images, 8 are used for training and 2 are used for validation.
    data_file = f"data/imagepatches_{train_args.patch_size}_comb.np"
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data_patches = np.load(f)
            val_patches = np.load(f)
    else:
        data_patches = np.moveaxis(extract_patches_2d(images, (train_args.patch_size, train_args.patch_size)), -1, 1). \
                                   reshape(-1, train_args.patch_size, train_args.patch_size)
        train_mean, train_std = np.mean(data_patches, axis=0), np.std(data_patches, axis=0)
        val_idx = np.linspace(1, data_patches.shape[0] - 1, int(len(data_patches)*0.2), dtype=int)
        train_idx = np.ones(len(data_patches), bool)
        train_idx[val_idx] = 0
        val_patches = data_patches[val_idx]
        data_patches = data_patches[train_idx]
        #data_patches = np.moveaxis(extract_patches_2d(images[:, :, :-2], (train_args.patch_size, train_args.patch_size)), -1, 1). \
        #    reshape(-1, train_args.patch_size, train_args.patch_size)
        #val_patches = np.moveaxis(extract_patches_2d(images[:, :, -2:], (train_args.patch_size, train_args.patch_size)), -1, 1). \
        #    reshape(-1, train_args.patch_size, train_args.patch_size)
        #with open(data_file, 'wb') as f:
        #    np.save(f, data_patches)
        #    np.save(f, val_patches)

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
        train_idx = np.linspace(1, data_patches.shape[0] - 1, train_args.train_samples, dtype=int)
        train_patches = data_patches[train_idx, :, :]
        #train_mean, train_std = np.mean(data_patches, axis=0), np.std(data_patches, axis=0)
        train_patches = (train_patches - train_mean) / train_std
        logging.info("Shape of training dataset: {}".format(train_patches.shape))

        # Designate patches to use for training & validation. This step will also normalize the data.
        val_patches = val_patches[np.linspace(1, val_patches.shape[0] - 1, train_args.val_samples, dtype=int), :, :]
        val_patches = (val_patches - train_mean) / train_std
        logging.info("Shape of validation dataset: {}".format(val_patches.shape))

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

        vi_opt = torch.optim.SGD(encoder.parameters(), lr=1e-3, #weight_decay=1e-4,
                                    momentum=0.9, nesterov=True)
        vi_scheduler = CycleScheduler(vi_opt, 1e-3, 
                                        n_iter=(train_args.epochs * train_patches.shape[0]) // train_args.batch_size,
                                        momentum=None, warmup_proportion=0.05)

        # Create core-set for prior
        if solver_args.prior_method == "coreset":
            mbed_file = np.load(solver_args.coreset_embed_path)
            if solver_args.coreset_feat == "pca":
                feat = mbed_file['pca_mbed']
            elif solver_args.coreset_feat == "isomap":
                feat = mbed_file['isomap_mbed']
            elif solver_args.coreset_feat == "wavelet":
                feat = mbed_file['wavelet_mbed']
            else:
                raise NotImplementedError

            logging.info(f"Building core-set using {solver_args.coreset_alg} with {solver_args.coreset_size} centroids...")
            if solver_args.coreset_alg == "kmedoids":
                kmedoid = KMedoids(n_clusters=solver_args.coreset_size, random_state=0).fit(feat)                
                encoder.coreset = torch.tensor(train_patches[kmedoid.medoid_indices_], device=default_device).reshape(solver_args.coreset_size , -1)
                encoder.coreset_labels = torch.tensor(kmedoid.labels_, device=default_device)   
                encoder.coreset_coeff = torch.tensor(mbed_file['codes'][kmedoid.medoid_indices_], device=default_device)              
            else:
                raise NotImplementedError
            logging.info(f"...core-set succesfully built.")
        
        torch.save({'model_state': encoder.state_dict()}, train_args.save_path + "encoderstate_epoch0.pt")

    # Initialize empty arrays for tracking learning data
    dictionary_saved = np.zeros((train_args.epochs, *dictionary.shape))
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
                    iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu, patch_idx)

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

            # Take gradient step on dictionaries
            generated_patch = dictionary @ b
            residual = patches - generated_patch
            #select_penalty = np.sqrt(np.sum(dictionary ** 2, axis=0)) > 1.5
            step = ((residual @ b.T) / train_args.batch_size) - 2*train_args.fnorm_reg*dictionary#*select_penalty

            dictionary += step_size * step
            # Normalize dictionaries. Required to prevent unbounded growth, Tikhonov regularisation also possible.
            if train_args.normalize:
                dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))
            # Calculate loss after gradient step
            epoch_loss[i] = 0.5 * np.sum((patches - dictionary @ b) ** 2) + solver_args.lambda_ * np.sum(np.abs(b))

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
                    iwae_loss, recon_loss, kl_loss, b_cu = encoder(patches_cu, dict_cu, patch_idx)
                    b_hat = b_cu.detach().cpu().numpy().T
                    b_true = FISTA(dictionary, patches, tau=solver_args.lambda_)

            # Compute and save loss
            epoch_true_recon[i] = 0.5 * np.sum((patches - dictionary @ b_true) ** 2)
            epoch_val_recon[i] = 0.5 * np.sum((patches - dictionary @ b_hat) ** 2)
            epoch_true_l1[i] = solver_args.lambda_ * np.sum(np.abs(b_true))
            epoch_val_l1[i] = solver_args.lambda_ * np.sum(np.abs(b_hat))
            epoch_iwae_loss[i], epoch_kl_loss[i]  = iwae_loss, kl_loss.mean()

        # Decay step-size
        step_size = step_size * train_args.lr_decay

        # Ramp up sigmoid for spike-slab
        if solver_args.prior_distribution == "concreteslab":
            encoder.temp *= 0.9
            if encoder.temp <= solver_args.temp_min:
                encoder.temp = solver_args.temp_min
        if solver_args.prior_method == "clf":
            encoder.clf_temp *= 0.9
            if encoder.clf_temp <= solver_args.clf_temp_min:
                encoder.clf_temp = solver_args.clf_temp_min

        # Save and print data from epoch
        train_time[j] = time.time() - init_time
        epoch_time = train_time[0] if j == 0 else train_time[j] - train_time[j - 1]
        train_loss[j] = np.sum(epoch_loss) / len(train_patches)
        val_recon[j], val_l1[j] = np.sum(epoch_val_recon) / len(val_patches), np.sum(epoch_val_l1) / len(val_patches)
        val_true_recon[j], val_true_l1[j] = np.sum(epoch_true_recon) / len(val_patches), np.sum(epoch_true_l1) / len(val_patches)
        val_iwae_loss[j], val_kl_loss[j]  = np.mean(epoch_iwae_loss), np.mean(epoch_kl_loss)
        val_kl_loss[j] = np.mean(epoch_kl_loss)
        coeff_est[j], coeff_true[j] = b_hat.T, b_true.T
        if solver_args.threshold and solver_args.solver == "VI":
            lambda_list[j] = encoder.lambda_.data.cpu().numpy()
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
                    val_kl_loss=val_kl_loss, coeff_est=coeff_est, coeff_true=coeff_true)
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
        compute_statistics(train_args.save_path, train_args.stat_im_count, train_args, solver_args)
