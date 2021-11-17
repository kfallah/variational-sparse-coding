import json
import os
from types import SimpleNamespace

import torch

import numpy as np
from matplotlib import pyplot as plt

from utils.solvers import FISTA
from model.lista import VIEncoderLISTA, LISTA
from model.vi_encoder import VIEncoder
from model.scheduler import CycleScheduler
from utils.dict_plotting import show_dict, show_phi_vid

code_count = 10000
real_images = True
start_with_dict = False
coadapt = False
epoch_load_list = ['FISTA', '0', '10', '50', '100', '200', '300']
epochs = 200
vsc_lr = 1e-2
initial_lr = 1e-2
lr_decay = 0.97

with open("results/VI_laplacian_coadapt/config.json") as json_data:
    config_data = json.load(json_data)
train_args = SimpleNamespace(**config_data['train'])
solver_args = SimpleNamespace(**config_data['solver'])

data_file = np.load("results/VI_laplacian_iwae/savefile_epoch199.npz")
phi = data_file['phi'][-1]

if not real_images:
    true_support = 30
    random_codes = np.zeros((code_count, phi.shape[1]))
    support = np.random.randint(0, high=phi.shape[1], size=(code_count, true_support))
    for i in range(code_count):
        random_codes[i, support[i]] = np.random.randn(true_support)
    random_images = random_codes @ phi.T
else:
    data_file = f"data/imagepatches_16.np"
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data_patches = np.load(f)
            val_patches = np.load(f)

    train_idx = np.linspace(1, data_patches.shape[0] - 1, code_count, dtype=int)
    random_images = data_patches[train_idx, :, :].reshape(-1, 16**2)
    random_images = random_images / np.linalg.norm(random_images, axis=1)[:, None]

    random_codes = np.zeros((code_count, phi.shape[1]))
    for i in range(code_count // 100):
        patches = random_images[i*100:(i+1)*100]
        random_codes[i*100:(i+1)*100] = FISTA(phi, patches.T, tau=solver_args.lambda_).T

save_dict = {}
recon_loss = {}
l1_loss = {}
true_coeffs = {}
est_coeffs = {}

for epoch_file in epoch_load_list:
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    train_patches = np.array(random_images)

    if epoch_file != 'FISTA':
        encoder = VIEncoder(train_args.patch_size, train_args.dict_size, solver_args).to('cuda')
        encoder.load_state_dict(torch.load(f"results/VI_laplacian_fixeddict/encoderstate_epoch{epoch_file}.pt")['model_state'])
        vi_opt = torch.optim.SGD(encoder.parameters(), lr=vsc_lr, #weight_decay=1e-4,
                            momentum=0.9, nesterov=True)
        vi_scheduler = CycleScheduler(vi_opt, vsc_lr, 
                                        n_iter=(epochs * train_patches.shape[0]) // train_args.batch_size,
                                        momentum=None, warmup_proportion=0.05)

    if start_with_dict:
        dictionary = np.array(phi)
    else:
        dictionary = np.random.randn(train_args.patch_size ** 2, train_args.dict_size)
        dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))

    iter_per_epoch = train_patches.shape[0] // train_args.batch_size
    save_dict[epoch_file] = np.zeros((iter_per_epoch*epochs, *dictionary.shape))
    recon_loss[epoch_file] = np.zeros(iter_per_epoch*epochs)
    l1_loss[epoch_file] = np.zeros(iter_per_epoch*epochs)
    true_coeffs[epoch_file] = np.zeros((epochs, train_args.batch_size, dictionary.shape[1]))
    est_coeffs[epoch_file] = np.zeros((epochs, train_args.batch_size, dictionary.shape[1]))
    step_size = initial_lr

    for j in range(epochs):
        # Shuffle training data-set
        np.random.shuffle(train_patches)
        for i in range(train_patches.shape[0] // train_args.batch_size):
            save_dict[epoch_file][j*iter_per_epoch + i] = np.array(dictionary)
            patches = train_patches[i * train_args.batch_size:(i + 1) * train_args.batch_size].reshape(train_args.batch_size, -1).T

            if epoch_file != 'FISTA':
                patches_cu = torch.tensor(patches.T).float().to('cuda')
                dict_cu = torch.tensor(dictionary, device='cuda').float()
                iwae_loss, recon, kl_loss, b_cu = encoder(patches_cu, dict_cu)
                b = b_cu.detach().cpu().numpy().T
                
                if coadapt:
                    vi_opt.zero_grad()
                    iwae_loss.backward()
                    vi_opt.step()
                    vi_scheduler.step()
            else:
                b = FISTA(dictionary, patches, tau=solver_args.lambda_)

            generated_patch = dictionary @ b
            residual = patches - generated_patch
            step = residual @ b.T
            dictionary += step_size * step

            # Normalize dictionaries. Required to prevent unbounded growth, Tikhonov regularisation also possible.
            dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))
            recon_loss[epoch_file][j*iter_per_epoch + i] = 0.5 * np.sum((patches - dictionary @ b) ** 2) 
            l1_loss[epoch_file][j*iter_per_epoch + i] = solver_args.lambda_ * np.sum(np.abs(b))

        b_true = FISTA(dictionary, patches, tau=solver_args.lambda_)
        true_coeffs[epoch_file][j] = b_true.T
        est_coeffs[epoch_file][j] = b.T

        step_size = step_size * lr_decay
        print("Epoch {} of {}, Avg Train Loss = {:.4f}".format(j + 1, epochs, (recon_loss[epoch_file][j:(j+1)*iter_per_epoch] + l1_loss[epoch_file][j:(j+1)*iter_per_epoch]).mean()))

epoch_names = [n if n == 'FISTA' else 'Epoch ' + n for n in epoch_load_list]
plt.figure(figsize=(8, 8))

for idx, run_name in enumerate(epoch_load_list):
    plt.semilogy(recon_loss[run_name] + l1_loss[run_name], label=epoch_names[idx]) 
plt.title("")
plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Training Objective", fontsize=16)
plt.legend(fontsize=16)
plt.savefig("results/perturbation/loss.png", bbox_inches='tight')
plt.close()

epoch_names = [n if n == 'FISTA' else 'Epoch ' + n for n in epoch_load_list]
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,8))

for idx, run_name in enumerate(epoch_load_list):
    ax[0].semilogy(recon_loss[run_name], label=epoch_names[idx]) 
    ax[1].semilogy(l1_loss[run_name], label=epoch_names[idx]) 

plt.title("")
plt.xlabel("Iteration", fontsize=16)
ax[0].set_ylabel("Recon Objective", fontsize=16)
ax[1].set_ylabel("L1 Objective", fontsize=16)
plt.legend(fontsize=12)
plt.savefig("results/perturbation/separate_loss.png", bbox_inches='tight')
plt.close()

for epoch_file in epoch_load_list:
    num_dictionaries = phi.shape[1]
    patch_size = int(np.sqrt(phi.shape[0]))
    fig = plt.figure(figsize=(12, 12))
    for i in range(num_dictionaries):
        plt.subplot(int(np.sqrt(num_dictionaries)), int(np.sqrt(num_dictionaries)), i + 1)
        dict_element = save_dict[epoch_file][-1, :, i].reshape(patch_size, patch_size)
        plt.imshow(dict_element, cmap='gray')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig(f"results/perturbation/final_dict_{epoch_file}.png", bbox_inches='tight')
    plt.close()

truecap_supp = {}
redundent_supp = {}
classification_acc = {}
active_supp = {}

for epoch_file in epoch_load_list:
    truecap_supp[epoch_file] = np.zeros(epochs)
    redundent_supp[epoch_file] = np.zeros(epochs)
    classification_acc[epoch_file] = np.zeros(epochs)
    active_supp[epoch_file] = np.zeros(epochs)

    for j in range(epochs):
        for i in range(train_args.batch_size):
            true_sup = np.nonzero(true_coeffs[epoch_file][j, i])[0]
            est_sup = np.nonzero(est_coeffs[epoch_file][j, i])[0]
            missed_support = np.setdiff1d(true_sup, est_sup)
            excess_support = np.setdiff1d(est_sup, true_sup)

            truecap_supp[epoch_file][j] += (1 - (len(missed_support) / dictionary.shape[1]))
            redundent_supp[epoch_file][j] += (len(excess_support) / dictionary.shape[1])
            classification_acc[epoch_file][j] += (dictionary.shape[1] - len(missed_support) - len(excess_support)) 
            active_supp[epoch_file][j] += len(est_sup)
        truecap_supp[epoch_file][j] /= (train_args.batch_size / 100.)
        redundent_supp[epoch_file][j] /= (train_args.batch_size / 100.)
        classification_acc[epoch_file][j] /= (train_args.batch_size * dictionary.shape[1] / 100.)
        active_supp[epoch_file][j] /= train_args.batch_size

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,12))
for idx, epoch_file in enumerate(epoch_load_list):
    ax[0, 0].plot(truecap_supp[epoch_file], label=epoch_names[idx], linewidth=3)
    ax[0, 1].plot(redundent_supp[epoch_file], label=epoch_names[idx], linewidth=3)
    ax[1, 0].plot(classification_acc[epoch_file], label=epoch_names[idx], linewidth=3)
    ax[1, 1].plot(active_supp[epoch_file], label=epoch_names[idx], linewidth=3)
ax[0, 0].set_ylim([-.1, 100.1])
ax[0, 0].set_title("True Captured Support", fontsize=14)
ax[0, 0].legend(fontsize=14)
ax[0, 1].set_ylim([-.1, 100.1])
ax[0, 1].set_title("Redundent Support", fontsize=14)
ax[1, 0].set_ylim([-.1, 100.1])
ax[1, 0].set_title("Classification Accuracy", fontsize=14)
ax[1, 1].set_title("Active Support", fontsize=14)
plt.savefig("results/perturbation/support_acc.png", bbox_inches='tight')
plt.close()