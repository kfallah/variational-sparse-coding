"""
Train auto-encoder using sparse features (i.e., Laplacian, Spike-and-Slab) to reconstruct CelebA images.

@Filename    train_vsc_celeba.py
@Author      Kion
@Created     01/03/22
"""
import argparse
import time
import os
import logging
import json, codecs
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
import torch

from utils.solvers import FISTA
from model.feature_enc import ConvDecoder
from model.vi_encoder import VIEncoder
from model.scheduler import CycleScheduler
from utils.data_loader import load_celeba
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

default_device = torch.device('cuda', train_args.device[0])
if not os.path.exists(train_args.save_path):
    os.makedirs(train_args.save_path)
    print("Created directory for figures at {}".format(train_args.save_path))
with open(train_args.save_path + '/config.json', 'wb') as f:
    json.dump(config_data, codecs.getwriter('utf-8')(f), ensure_ascii=False, indent=2)
logging.basicConfig(filename=os.path.join(train_args.save_path, 'training.log'), 
                    filemode='w', level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

if __name__ == "__main__":
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    # LOAD DATASET #
    train_loader, test_loader = load_celeba("./data/", train_args)

    # INITIALIZE 
    decoder = ConvDecoder(train_args.dict_size, 3).to(default_device)
    params = list(decoder.parameters())

    if solver_args.solver == "VI":
        encoder = VIEncoder(16, train_args.dict_size, solver_args).to(default_device)   
        params += list(encoder.parameters())
        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, train_args.save_path + "modelstate_epoch0.pt")
    else:
        torch.save({'decoder': decoder.state_dict()}, train_args.save_path + "modelstate_epoch0.pt")

    opt = torch.optim.SGD(params, lr=train_args.lr, weight_decay=train_args.weight_decay,
                          momentum=0.9, nesterov=True)
    scheduler = CycleScheduler(opt, train_args.lr,  n_iter=train_args.epochs * len(train_loader),
                               momentum=None, warmup_proportion=0.05) 

    # Initialize empty arrays for tracking learning data
    lambda_list = np.zeros((train_args.epochs, train_args.dict_size))
    coeff_est = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
    train_loss, val_recon, val_l1 = np.zeros(train_args.epochs), np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    val_iwae_loss, val_kl_loss = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    train_time = np.zeros(train_args.epochs)

    # TRAIN MODEL #
    init_time = time.time()
    for j in range(train_args.epochs):
        epoch_loss = np.zeros(len(train_loader))
        if solver_args.solver == "VI":
            encoder.train()
        decoder.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(default_device)

            # Infer coefficients
            if solver_args.solver == "FISTA":
                # TODO: Transfer FISTA via PyTorch
                b_cu = FISTA(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "VI":
                iwae_loss, recon_loss, kl_loss, b_cu = encoder(x, decoder)
            opt.zero_grad()
            iwae_loss.backward()
            opt.step()
            scheduler.step()

            # Calculate loss after gradient step
            epoch_loss[i] = 0.5 * recon_loss.mean().item() + solver_args.lambda_ * torch.sum(torch.abs(b_cu)).detach().cpu().numpy()

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
                if (len(train_loader)*j + i) >= 1500:
                    encoder.warmup += 2e-4
                    if encoder.warmup >= 1.0:
                        encoder.warmup = 1.0

        # Test reconstructed or uncompressed dictionary on validation data-set
        epoch_val_recon = np.zeros(len(test_loader))
        epoch_val_l1 = np.zeros(len(test_loader))
        epoch_iwae_loss = np.zeros(len(test_loader))
        epoch_kl_loss = np.zeros(len(test_loader))
        if solver_args.solver == "VI":
            encoder.eval()
        decoder.eval()
        for i, (x, y) in enumerate(test_loader):
            # Load next batch of validation patches
            x = x.to(default_device)

            # Infer coefficients
            if solver_args.solver == "FISTA":
                b_hat = FISTA(dictionary, patches, tau=solver_args.lambda_)
                b_true = np.array(b_hat)
                iwae_loss, kl_loss = 0., np.array(0.)
            elif solver_args.solver == "VI":
                with torch.no_grad():
                    iwae_loss, recon_loss, kl_loss, b_cu = encoder(x, decoder)
                b_hat = b_cu.detach().cpu().numpy()

            # Compute and save loss
            epoch_val_recon[i] = recon_loss.mean().item()
            epoch_val_l1[i] = torch.sum(torch.abs(b_cu)).detach().cpu().numpy()
            epoch_iwae_loss[i], epoch_kl_loss[i]  = iwae_loss.item(), kl_loss.mean().item()

        # Save and print data from epoch
        train_time[j] = time.time() - init_time
        epoch_time = train_time[0] if j == 0 else train_time[j] - train_time[j - 1]
        train_loss[j] = np.sum(epoch_loss) / len(train_loader.dataset)
        val_recon[j], val_l1[j] = np.sum(epoch_val_recon) / len(test_loader.dataset), np.sum(epoch_val_l1) / len(test_loader.dataset)
        val_iwae_loss[j], val_kl_loss[j]  = np.mean(epoch_iwae_loss), np.mean(epoch_kl_loss)
        coeff_est[j] = b_cu.detach().cpu().numpy()
        if solver_args.threshold and solver_args.solver == "VI":
            lambda_list[j] = encoder.lambda_.data.mean(dim=0).cpu().numpy()
        else:
            lambda_list[j] = np.ones(train_args.dict_size) * -1

        if solver_args.debug:
            print_debug(train_args, b_hat, b_hat)
            logging.info("Mean lambda value: {:.3E}".format(lambda_list[j].mean()))
            logging.info("Est IWAE loss: {:.3E}".format(val_iwae_loss[j]))
            logging.info("Est KL loss: {:.3E}".format(val_kl_loss[j]))
            logging.info("Est total loss: {:.3E}".format(val_recon[j] + solver_args.lambda_ * val_l1[j]))

        if j < 10 or (j + 1) % train_args.save_freq == 0 or (j + 1) == train_args.epochs:
            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(14, 8))
            with torch.no_grad():
                x_hat = decoder(b_cu)
                for im_idx in range(5):
                    ax[0, im_idx].imshow(x[im_idx].permute(1, 2, 0).detach().cpu().numpy())
                    ax[1, im_idx].imshow(x_hat[im_idx].permute(1, 2, 0).detach().cpu().numpy())
                plt.savefig(train_args.save_path + f"recon_image_epoch{j+1}.png", bbox_inches='tight')
                plt.close()

            np.savez_compressed(train_args.save_path + f"train_savefile.npz",
                    lambda_list=lambda_list, time=train_time, train=train_loss, val_recon=val_recon, 
                    val_l1=val_l1, val_iwae_loss=val_iwae_loss, val_kl_loss=val_kl_loss, coeff_est=coeff_est)
            if solver_args.solver == "VI":
                torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, 
                           train_args.save_path + f"modelstate_epoch{j+1}.pt")
            else:
                torch.save({'decoder': decoder.state_dict()}, train_args.save_path + f"modelstate_epoch{j+1}.pt")
        logging.info("Epoch {} of {}, Avg Train Loss = {:.4E}, Avg Val Loss = {:.4E}, Time = {:.0f} secs".format(j + 1,
                                                                                                          train_args.epochs,
                                                                                                          train_loss[j],
                                                                                                          val_recon[j] + solver_args.lambda_ * val_l1[j],
                                                                                                          epoch_time))
        logging.info("\n")