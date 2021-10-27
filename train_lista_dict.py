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

import numpy as np
import scipy.io
from sklearn.feature_extraction.image import extract_patches_2d

import torch
import torch.nn.functional as F

from utils.dict_plotting import show_dict
from utils.solvers import FISTA, ADMM
from model.encoder import MLPEncoderLISTA, sample_laplacian, sample_gaussian
from model.scheduler import CycleScheduler

# PARSE COMMAND LINE ARGUMENTS #
parser = argparse.ArgumentParser(description='Run sparse dictionary learning with compressed images.')
parser.add_argument('-S', '--solver', default='FISTA', choices=['VI', 'FISTA', 'ADMM'],
                    help="Solver used to find sparse coefficients")
parser.add_argument('-b', '--batch_size', default=100, type=int, help="Batch size")
parser.add_argument('-N', '--dict_count', default=256, type=int, help="Dictionary count")
parser.add_argument('-P', '--patch_size', default=16, type=int, help="Patch size")
parser.add_argument('-R', '--l1_penalty', default=1e-1, type=float, help="L1 regularizer constant")
parser.add_argument('-e', '--num_epochs', default=200, type=int, help="Number of epochs")
parser.add_argument('-T', '--train_samples', default=60000, type=int, help="Number of training samples to use")
parser.add_argument('-V', '--val_samples', default=15000, type=int, help="Number of validation samples to use")
parser.add_argument('-s', '--spread_prior', default=0.1, type=float, help="Weighting for spread prior")
parser.add_argument('-k', '--kl_weight', default=0.1, type=float, help="Weighting on KL term")
parser.add_argument('-p', '--prior', default='laplacian', choices=['laplacian', 'gaussian', 'spikeslab'],
                    help="Prior for encoder")
parser.add_argument('-L', '--learning_rate', default=0.02, type=float, help="Default initial learning rate")
parser.add_argument('-d', '--decay', default=.98, type=float, help="Default multiplicative learning rate decay")

# PARSE ARGUMENTS #
args = parser.parse_args()
solver = args.solver
batch_size = args.batch_size
num_dictionaries = args.dict_count
patch_size = args.patch_size
tau = args.l1_penalty
num_epochs = args.num_epochs
train_samples = args.train_samples
val_samples = args.val_samples
learning_rate = args.learning_rate
decay = args.decay
spread_prior = args.spread_prior
kl_weight = args.kl_weight
prior = args.prior
debug = True

save_dir = "results/" + time.strftime("%m-%d-%Y") + f"_{solver}_{kl_weight}kl_{prior}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for figures at {}".format(save_dir))

if __name__ == "__main__":
    # LOAD DATA #
    data_matlab = scipy.io.loadmat('./data/whitened_images.mat')
    images = np.ascontiguousarray(data_matlab['IMAGES'])

    # Extract patches using SciKit-Learn. Out of 10 images, 8 are used for training and 2 are used for validation.
    data_file = f"data/imagepatches_{patch_size}.np"
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data_patches = np.load(f)
            val_patches = np.load(f)
    else:
        data_patches = np.moveaxis(extract_patches_2d(images[:, :, :-2], (patch_size, patch_size)), -1, 1). \
            reshape(-1, patch_size, patch_size)
        val_patches = extract_patches_2d(images[:, :, -2], (patch_size, patch_size))
        with open(data_file, 'wb') as f:
            np.save(f, data_patches)
            np.save(f, val_patches)

    # Designate patches to use for training, validation, and correlation (only for compressed dictionaries). This
    # step will also normalize the data.
    val_patches = val_patches[np.linspace(1, val_patches.shape[0] - 1, val_samples, dtype=int), :, :]
    val_patches = val_patches / np.linalg.norm(val_patches.reshape(-1, patch_size ** 2), axis=1)[:, None, None]
    print("Shape of validation dataset: {}".format(val_patches.shape))

    train_idx = np.linspace(1, data_patches.shape[0] - 1, train_samples, dtype=int)
    train_patches = data_patches[train_idx, :, :]
    train_patches = train_patches / np.linalg.norm(train_patches.reshape(-1, patch_size ** 2), axis=1)[:, None, None]
    print("Shape of training dataset: {}".format(train_patches.shape))

    # INITIALIZE TRAINING PARAMETERS #
    dictionary = np.random.randn(patch_size ** 2, num_dictionaries)
    dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))
    step_size = learning_rate

    if solver == "VI":
        encoder = MLPEncoderLISTA(patch_size, num_dictionaries, tau).to('cuda')
        vi_opt = torch.optim.SGD(encoder.parameters(), lr=1e-3, weight_decay=1e-4,
                                 momentum=0.9, nesterov=True)
        vi_scheudler = CycleScheduler(vi_opt, 1e-3, 
                                      n_iter=(num_epochs * train_patches.shape[0]) // batch_size,
                                      momentum=None, warmup_proportion=0.05)

    # Initialize empty arrays for tracking learning data
    dictionary_saved = np.zeros((num_epochs, *dictionary.shape))
    coeff_true = np.zeros((num_epochs, batch_size, num_dictionaries))
    coeff_est = np.zeros((num_epochs, batch_size, num_dictionaries))
    train_loss = np.zeros(num_epochs)
    val_true_recon = np.zeros(num_epochs)
    val_recon = np.zeros(num_epochs)
    val_true_l1 = np.zeros(num_epochs)
    val_l1 = np.zeros(num_epochs)
    val_kl_loss = np.zeros(num_epochs)
    train_time = np.zeros(num_epochs)

    # TRAIN MODEL #
    init_time = time.time()
    for j in range(num_epochs):
        epoch_loss = np.zeros(train_patches.shape[0] // batch_size)
        # Shuffle training data-set
        np.random.shuffle(train_patches)
        for i in range(train_patches.shape[0] // batch_size):
            patches = train_patches[i * batch_size:(i + 1) * batch_size].reshape(batch_size, -1).T

            # Infer coefficients
            if solver == "FISTA":
                b = FISTA(dictionary, patches, tau=tau)
            elif solver == "ADMM":
                b = ADMM(dictionary, patches, tau=tau)
            elif solver == "VI":
                var_samples = 1
                grad_list = []
                for i in range(var_samples):
                    patches_cu = torch.tensor(patches.T).float().to('cuda')
                    dict_cu = torch.tensor(dictionary, device='cuda').float()
                    b_scale, b_cu = encoder(patches_cu, dict_cu)

                    recon_loss = F.mse_loss(patches_cu, (b_cu @ dict_cu.T), reduction='mean')
                    kl_loss = 0.5 * (b_scale - torch.log(b_scale) - 1.0).mean()

                    vi_opt.zero_grad()
                    (recon_loss + kl_weight * kl_loss).backward()
                    #model_grad = [param.grad.data.reshape(-1) for param in encoder.parameters()]
                    #model_grad = torch.cat(model_grad)
                    #grad_list.append(model_grad.detach().cpu().numpy())

                #grad_list = np.stack(grad_list)
                #if i == 0:
                #    print("VARIANCE:")
                #    print(np.var(grad_list, axis=0).mean())
                vi_opt.step()
                vi_scheudler.step()
                #b = b_cu.detach().cpu().numpy().T
                b = FISTA(dictionary, patches, tau=tau)

            # Take gradient step on dictionaries
            generated_patch = dictionary @ b
            residual = patches - generated_patch
            step = residual @ b.T
            dictionary += step_size * step

            # Normalize dictionaries. Required to prevent unbounded growth, Tikhonov regularisation also possible.
            dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))
            # Calculate loss after gradient step
            epoch_loss[i] = 0.5 * np.sum((patches - dictionary @ b) ** 2) + tau * np.sum(np.abs(b))

        # Test reconstructed or uncompressed dictionary on validation data-set
        epoch_true_recon = np.zeros(val_patches.shape[0] // batch_size)
        epoch_val_recon = np.zeros(val_patches.shape[0] // batch_size)
        epoch_true_l1 = np.zeros(val_patches.shape[0] // batch_size)
        epoch_val_l1 = np.zeros(val_patches.shape[0] // batch_size)
        epoch_kl_loss = np.zeros(val_patches.shape[0] // batch_size)

        for i in range(val_patches.shape[0] // batch_size):
            # Load next batch of validation patches
            patches = val_patches[i * batch_size:(i + 1) * batch_size].reshape(batch_size, -1).T

            # Infer coefficients
            if solver == "FISTA":
                b_hat = FISTA(dictionary, patches, tau=tau)
            elif solver == "ADMM":
                b_hat = ADMM(dictionary, patches, tau=tau)
            elif solver == "VI":
                with torch.no_grad():
                    patches_cu = torch.tensor(patches.T).float().to('cuda')
                    dict_cu = torch.tensor(dictionary, device='cuda').float()
                    b_scale, b_cu = encoder(patches_cu, dict_cu)
                    b_hat = b_cu.detach().cpu().numpy().T
                    b_true = FISTA(dictionary, patches, tau=tau)

            # Compute and save loss
            epoch_true_recon[i] = 0.5 * np.sum((patches - dictionary @ b_true) ** 2)
            epoch_val_recon[i] = 0.5 * np.sum((patches - dictionary @ b_hat) ** 2)
            epoch_true_l1[i] = tau * np.sum(np.abs(b_true))
            epoch_val_l1[i] = tau * np.sum(np.abs(b_hat))
            epoch_kl_loss[i] = (0.5 * (b_scale - torch.log(b_scale) - 1.0).mean()).item()

        # Decay step-size
        step_size = step_size * decay

        # Save and print data from epoch
        train_time[j] = time.time() - init_time
        epoch_time = train_time[0] if j == 0 else train_time[j] - train_time[j - 1]
        train_loss[j] = np.mean(epoch_loss)
        val_true_recon[j] = np.mean(epoch_true_recon)
        val_recon[j] = np.mean(epoch_val_recon)
        val_true_l1[j] = np.mean(epoch_true_l1)
        val_l1[j] = np.mean(epoch_val_l1)
        val_kl_loss[j] = np.mean(epoch_kl_loss)
        dictionary_saved[j] = dictionary

        if debug:
            count_nz = np.zeros(num_dictionaries + 1, dtype=int)
            coeff_nz = np.count_nonzero(b_hat.T, axis=0)
            nz_tot = np.count_nonzero(coeff_nz)
            total_nz = np.count_nonzero(b_hat.T, axis=1)
            true_total_nz = np.count_nonzero(b_true.T, axis=1)

            for z in range(len(total_nz)):
                count_nz[total_nz[z]] += 1
            mean_coeff = b_hat.T[b_hat.T > 0].mean()
            true_coeff = b_true.T[b_true.T > 0].mean()

            supp_diff = 0
            for k in range(len(b_hat.T)):
                est_supp = np.where(b_hat.T[k] >= 1e-6)[0]
                true_supp = np.where(b_true.T[k] >= 1e-6)[0]
                supp_diff += len(np.setdiff1d(true_supp, est_supp))
            supp_diff /= len(b_hat.T)
            coeff_est[j] = b_hat.T
            coeff_true[j] = b_true.T

            #print("Non-zero elements per bin: {}".format(count_nz))
            print("Non-zero by coefficient #: {}".format(nz_tot))
            print(f"Mean non-zero coeff: {total_nz.mean():.3f}")
            print(f"Mean true non-zero coeff: {true_total_nz.mean():.3f}")
            print(f"Mean coeff value: {mean_coeff}")
            print(f"Mean true coeff value: {true_coeff}")
            print("Coeff gap: {:.3E}".format(np.linalg.norm(b_hat - b_true)))
            print("Avg support diff for coeff: {:.3f}".format(supp_diff))
            print("LISTA mean spread: {:.3E}".format(b_scale.detach().mean()))
            print(f"Est recon: {val_recon[j]:.3E}")
            print(f"True recon: {val_true_recon[j]:.3E}")
            print("Est KL loss: {:.3E}".format(val_kl_loss[j]))
            print("Est total loss: {:.3E}".format(val_recon[j] + val_l1[j]))
            print("True total loss: {:.3E}".format(val_l1[j] + val_true_l1[j]))

        if j % 10 == 0 or j == num_epochs-1:
            show_dict(dictionary, save_dir + f"dictionary_epoch{j}.png")
            np.savez_compressed(save_dir + f"savefile_epoch{j}.npz",
                    phi=dictionary_saved, time=train_time, train=train_loss, 
                    val_true_recon=val_true_recon, val_recon=val_recon, 
                    val_l1=val_l1, val_true_l1=val_true_l1,
                    val_kl_loss=val_kl_loss, coeff_est=coeff_est, coeff_true=coeff_true)
        print("Epoch {} of {}, Avg Train Loss = {:.4f}, Avg Val Loss = {:.4f}, Time = {:.0f} secs".format(j + 1,
                                                                                                          num_epochs,
                                                                                                          train_loss[j],
                                                                                                          val_recon[j] + val_l1[j],
                                                                                                          epoch_time))
        print()
