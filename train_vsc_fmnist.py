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
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import torch.distributed as dist
import torch.multiprocessing as mp

from model.feature_enc import ConvDecoder
from model.vi_encoder import VIEncoder
from model.util import FISTA_pytorch, frange_cycle_linear
from model.scheduler import CycleScheduler
from utils.data_loader import load_fmnist
from utils.util import *

def train(gpu, train_args, solver_args):

    train_args.rank = gpu
    """
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=train_args.world_size,
    	rank=train_args.rank
    )
    """

    if train_args.rank == 0:
        logging.basicConfig(filename=os.path.join(train_args.save_path, 'training.log'), 
                            filemode='w', level=logging.DEBUG)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed(train_args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # LOAD DATASET #
    train_loader, test_loader = load_fmnist("./data/", train_args, distributed=False)

    # INITIALIZE 
    torch.cuda.set_device(train_args.device[0])
    default_device = torch.device('cuda', train_args.device[0])

    decoder = ConvDecoder(train_args.dict_size, 1, im_size=28).to(default_device)
    #decoder = nn.parallel.DistributedDataParallel(decoder, device_ids=[gpu])
    scaler = GradScaler(enabled=train_args.amp)

    if solver_args.solver == "VI":
        encoder = VIEncoder(16, train_args.dict_size, solver_args, input_size=(1, 28, 28)).to(default_device)
        #encoder = nn.parallel.DistributedDataParallel(encoder, device_ids=[gpu])

        opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                lr=train_args.lr, betas=(0.5, 0.999), weight_decay=train_args.weight_decay) 
        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, train_args.save_path + "modelstate_epoch0.pt")

        if solver_args.prior_distribution == "laplacian":
            encoder.ramp_hyperparams()
    else:
        #opt = torch.optim.SGD(decoder.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay,
        #                      momentum=0.9, nesterov=True)
        lambda_warmup = 1e-2
        opt = torch.optim.Adam(decoder.parameters(), lr=train_args.lr, betas=(0.5, 0.999), 
                               weight_decay=train_args.weight_decay) 
        torch.save({'decoder': decoder.state_dict()}, train_args.save_path + "modelstate_epoch0.pt")

    scheduler = CycleScheduler(opt, train_args.lr,  n_iter=train_args.epochs * len(train_loader),
                               momentum=None, warmup_proportion=0.05) 

    if solver_args.kl_schedule:
        if solver_args.theshold_learn or solver_args.prior_distribution == "laplacian":
            kl_schedule = np.ones(len(train_loader)*train_args.epochs) * solver_args.kl_weight
            ramp = np.linspace(1e-6, 1., int(len(kl_schedule)*0.10))
            kl_schedule[:len(ramp)] *= ramp
        else:
            kl_schedule = frange_cycle_linear(len(train_loader)*train_args.epochs, start=1e-9, 
                                                stop=solver_args.kl_weight,  n_cycle=4, ratio=0.5)

    # Initialize empty arrays for tracking learning data
    lambda_list = np.zeros((train_args.epochs, train_args.dict_size))
    coeff_est = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
    val_recon, val_l1 = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    val_iwae_loss, val_kl_loss = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    train_time = np.zeros(train_args.epochs)

    # TRAIN MODEL #
    init_time = time.time()
    for j in range(train_args.epochs):
        if solver_args.solver == "VI":
            encoder.train()
        decoder.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda(non_blocking=True)
            torch.cuda.synchronize()

            with autocast(enabled=train_args.amp):
                # Infer coefficients
                if solver_args.solver == "FISTA":
                    decoder.eval()
                    _, b_cu = FISTA_pytorch(x, decoder, train_args.dict_size, 
                                            lambda_warmup*solver_args.lambda_, max_iter=1500, tol=1e-6, 
                                            clip_grad=solver_args.clip_grad, device=default_device)
                    lambda_warmup += 1e-3
                    if lambda_warmup >= 1.0:
                        lambda_warmup = 1.0
                    decoder.train()
                    x_hat = decoder(b_cu)
                    iwae_loss = F.mse_loss(x_hat, x) 
                elif solver_args.solver == "VI":
                    if solver_args.kl_schedule:
                        solver_args.kl_weight = kl_schedule[len(train_loader)*j + i]
                    iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(x, decoder)

                opt.zero_grad()
                scaler.scale(iwae_loss).backward()
                scaler.step(opt)
                scaler.update()                
                scheduler.step()

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
                if (len(train_loader)*j + i) >= 500:
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

            with autocast(enabled=train_args.amp):
                # Infer coefficients
                if solver_args.solver == "FISTA":
                    (recon_loss, _, _), b_cu = FISTA_pytorch(x, decoder, train_args.dict_size, 
                                                            lambda_warmup*solver_args.lambda_, max_iter=1500, tol=1e-6, 
                                                            clip_grad=solver_args.clip_grad, device=default_device)
                    with torch.no_grad():
                        x_hat = decoder(b_cu)
                        recon_loss = F.mse_loss(x_hat, x).item()
                    iwae_loss, kl_loss = torch.tensor(-1.), torch.tensor(-1.)
                    b_select = b_cu.detach()
                    b_hat = b_cu.detach().cpu().numpy()
                elif solver_args.solver == "VI":
                    with torch.no_grad():
                        iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(x, decoder)
                        recon_loss = recon_loss.mean().item()
                        sample_idx = torch.distributions.categorical.Categorical(weight).sample().detach()
                        b_select = b_cu[torch.arange(len(b_cu)), sample_idx].detach()
                        b_hat = b_select.cpu().numpy()

            # Compute and save loss
            epoch_val_recon[i] = recon_loss
            epoch_val_l1[i] = np.sum(np.abs(b_hat))
            epoch_iwae_loss[i], epoch_kl_loss[i]  = iwae_loss.item(), kl_loss.mean().item()

        # Save and print data from epoch
        train_time[j] = time.time() - init_time
        epoch_time = train_time[0] if j == 0 else train_time[j] - train_time[j - 1]
        val_recon[j], val_l1[j] = np.sum(epoch_val_recon) / len(test_loader.dataset), np.sum(epoch_val_l1) / len(test_loader.dataset)
        val_iwae_loss[j], val_kl_loss[j]  = np.mean(epoch_iwae_loss), np.mean(epoch_kl_loss)
        coeff_est[j] = b_hat
        if solver_args.threshold and solver_args.solver == "VI":
            lambda_list[j] = encoder.lambda_.data.mean(dim=(0,1)).cpu().numpy()
        else:
            lambda_list[j] = np.ones(train_args.dict_size) * -1

        if train_args.rank == 0:
            if solver_args.debug:
                print_debug(train_args, b_hat, b_hat)
                for param_group in opt.param_groups:
                    logging.info(param_group['lr'])
                logging.info("Mean lambda value: {:.3E}".format(lambda_list[j].mean()))
                logging.info("Est IWAE loss: {:.3E}".format(val_iwae_loss[j]))
                logging.info("Est KL loss: {:.3E}".format(val_kl_loss[j]))
                logging.info("Est total loss: {:.3E}".format(val_recon[j] + solver_args.lambda_ * val_l1[j]))

            if j < 10 or (j + 1) % train_args.save_freq == 0 or (j + 1) == train_args.epochs:
                fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(14, 8))
                with torch.no_grad():
                    x_hat = decoder(b_select)
                    for im_idx in range(5):
                        ax[0, im_idx].imshow(x[im_idx].permute(1, 2, 0).detach().cpu().numpy())
                        ax[1, im_idx].imshow(x_hat[im_idx].permute(1, 2, 0).detach().cpu().numpy())
                    plt.savefig(train_args.save_path + f"recon_image_epoch{j+1}.png", bbox_inches='tight')
                    plt.close()

                np.savez_compressed(train_args.save_path + f"train_savefile.npz",
                        lambda_list=lambda_list, time=train_time, val_recon=val_recon, 
                        val_l1=val_l1, val_iwae_loss=val_iwae_loss, val_kl_loss=val_kl_loss, coeff_est=coeff_est)
                if solver_args.solver == "VI":
                    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, 
                            train_args.save_path + f"modelstate_epoch{j+1}.pt")
                else:
                    torch.save({'decoder': decoder.state_dict()}, train_args.save_path + f"modelstate_epoch{j+1}.pt")
                    
            logging.info("Epoch {} of {}, Val Recon: {:.3E}, Val KL: {:.3E}, Val SC: {:.3E}, Time = {:.0f} secs"\
                        .format(j + 1, train_args.epochs, val_recon[j],  val_kl_loss[j], \
                                val_recon[j] + solver_args.lambda_ * val_l1[j], epoch_time))
            logging.info("\n")

if __name__ == "__main__":
    # Load arguments for training via config file input to CLI #
    parser = argparse.ArgumentParser(description='Variational Sparse Coding')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file for training.')
    args = parser.parse_args()
    with open(args.config) as json_data:
        config_data = json.load(json_data)
    train_args = SimpleNamespace(**config_data['train'])
    solver_args = SimpleNamespace(**config_data['solver'])

    if not os.path.exists(train_args.save_path):
        os.makedirs(train_args.save_path)
        print("Created directory for figures at {}".format(train_args.save_path))
    with open(train_args.save_path + '/config.json', 'wb') as f:
        json.dump(config_data, codecs.getwriter('utf-8')(f), ensure_ascii=False, indent=2)


    train(0, train_args, solver_args)
    """
    world_size = len(train_args.device)
    train_args.world_size = world_size
    os.environ['MASTER_ADDR'] = '143.215.148.217'
    os.environ['MASTER_PORT'] = str(8888 + train_args.device[0])
    mp.spawn(train, nprocs=len(train_args.device), args=(train_args,solver_args))
    """