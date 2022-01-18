import logging
import os

from matplotlib import pyplot as plt
import ot

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.autoencoder import Encoder, Decoder
from model.scheduler import CycleScheduler

save_path = "sae_results/MNIST_run1/"
default_device = torch.device('cuda:0')
eps = 1e1
epochs = 100
beta = 100

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("Created directory for figures at {}".format(save_path))
logging.basicConfig(filename=os.path.join(save_path, 'training.log'), 
                    filemode='w', level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

train_dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                          shuffle=True, num_workers=8)

test_dataset = torchvision.datasets.MNIST('./data/', train=False, download=False,
                                          transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                          shuffle=True, num_workers=8)

encoder = Encoder(1, 128, 20).to(default_device)
decoder = Decoder(1, 128, 20).to(default_device)

opt = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-2, #weight_decay=1e-4,
                            momentum=0.9, nesterov=True)
scheduler = CycleScheduler(opt, 1e-2, 
                           n_iter=(200 * len(train_loader)) // 256,
                           momentum=None, warmup_proportion=0.05)

val_recon_loss = torch.zeros(epochs)
val_sinkhorn_loss = torch.zeros(epochs)
logging.info("Starting training...")

for i in range(epochs):
    encoder.train()
    decoder.train()
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(default_device), y.to(default_device)

        # Compute Sinkhorn cost
        z_hat = encoder(x)
        z = torch.randn_like(z_hat)
        sinkhorn_loss = ot.bregman.empirical_sinkhorn_divergence(z_hat, z, eps, to_numpy=False)

        # Compute reconstruction loss
        x_hat = decoder(z_hat)
        recon_loss = F.mse_loss(x_hat, x)

        opt.zero_grad()
        total_loss = recon_loss + beta * sinkhorn_loss
        total_loss.backward()
        opt.step()
        scheduler.step()
        logging.info(f"Iter {idx+1} of {len(train_loader)} recon: {recon_loss.item():.3E}, sinkhorn: {sinkhorn_loss.item():.3E}")

    val_recon_epoch = torch.zeros(len(test_loader))
    val_sinkhorn_epoch = torch.zeros(len(test_loader))
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(default_device), y.to(default_device)

            # Compute Sinkhorn cost
            z_hat = encoder(x)
            z = torch.randn_like(z_hat)
            sinkhorn_loss = ot.bregman.empirical_sinkhorn_divergence(z_hat, z, eps, to_numpy=False)

            # Compute reconstruction loss
            x_hat = decoder(z_hat)
            recon_loss = F.mse_loss(x_hat, x)

            val_recon_epoch[idx] = recon_loss.item()
            val_sinkhorn_epoch[idx] = sinkhorn_loss.item()

    val_recon_loss[i] = recon_loss.mean()
    val_sinkhorn_loss[i] = sinkhorn_loss.mean()
    logging.info(f"Epoch {i+1} of {epochs} recon: {val_recon_loss[i]:.3E}, sinkhorn: {val_sinkhorn_loss[i]:.3E}")
    if i <= 10 or (i + 1) % 20 == 0:
        torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'val_recon_loss': val_recon_loss,
                'val_sinkhorn_loss': val_sinkhorn_loss
                }, save_path + f"modelstate_epoch{i+1}.pt")

        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(14, 8))
        for im_idx in range(5):
            ax[0, im_idx].imshow(x[im_idx].reshape(28, 28, 1).detach().cpu().numpy(), cmap='gray')
            ax[1, im_idx].imshow(x_hat[im_idx].reshape(28, 28, 1).detach().cpu().numpy(), cmap='gray')
        plt.savefig(f"{save_path}recon_image_epoch{i+1}.png", bbox_inches='tight')
        plt.close()





