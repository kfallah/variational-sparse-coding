import os
import logging

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import scipy.io
from sklearn.feature_extraction.image import extract_patches_2d

def load_fmnist(path, train_args, distributed=False, imsize=28):
    train_data = torchvision.datasets.FashionMNIST(path, train=True, download=True,
                                    transform=transforms.Compose([    
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ]))
    train_data = torch.utils.data.Subset(train_data, torch.arange(train_args.train_samples))
    
    test_data = torchvision.datasets.FashionMNIST(path, train=False, download=False,
                                    transform=transforms.Compose([    
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ]))
    test_data = torch.utils.data.Subset(test_data, torch.arange(train_args.val_samples))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_data, num_replicas=train_args.world_size, rank=train_args.rank,
                            shuffle=True, seed=train_args.seed
        )

        test_sampler = torch.utils.data.distributed.DistributedSampler(
                            test_data, num_replicas=train_args.world_size, rank=train_args.rank,
                            shuffle=False, seed=train_args.seed
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_args.batch_size,
                                                shuffle=False, num_workers=8, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_args.batch_size,
                                                shuffle=False, num_workers=8, sampler=test_sampler,
                                                drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_args.batch_size,
                                                shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_args.batch_size,
                                                shuffle=False, num_workers=8, drop_last=True)
    logging.info(f"Dataset loaded with {len(train_data)} training and {len(test_data)} test images")
    return train_loader, test_loader

def load_celeba(path, train_args, distributed=False, imsize=64):
    train_data = torchvision.datasets.CelebA(path, split='train', target_type = 'attr',
                                    download=False,
                                    transform=transforms.Compose([    
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(140),
                                        transforms.Resize(imsize),
                                        transforms.ToTensor(),
                                    ]))
    train_data = torch.utils.data.Subset(train_data, torch.arange(train_args.train_samples))
    
    test_data = torchvision.datasets.CelebA(path, split='test', target_type = 'attr',
                                        download=False,
                                        transform=transforms.Compose([
                                        transforms.CenterCrop(140),
                                        transforms.Resize(imsize),
                                        transforms.ToTensor(),
                                    ]))
    test_data = torch.utils.data.Subset(test_data, torch.arange(train_args.val_samples))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_data, num_replicas=train_args.world_size, rank=train_args.rank,
                            shuffle=True, seed=train_args.seed
        )

        test_sampler = torch.utils.data.distributed.DistributedSampler(
                            test_data, num_replicas=train_args.world_size, rank=train_args.rank,
                            shuffle=False, seed=train_args.seed
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_args.batch_size,
                                                shuffle=False, num_workers=8, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_args.batch_size,
                                                shuffle=False, num_workers=8, sampler=test_sampler,
                                                drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_args.batch_size,
                                                shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_args.batch_size,
                                                shuffle=False, num_workers=8, drop_last=True)
    logging.info(f"Dataset loaded with {len(train_data)} training and {len(test_data)} test images")
    return train_loader, test_loader

def load_whitened_images(train_args, dictionary):

    # LOAD DATA #
    data_matlab = scipy.io.loadmat('./data/whitened_images.mat')
    images = np.ascontiguousarray(data_matlab['IMAGES'])

    # Extract patches using SciKit-Learn. Out of 10 images, 8 are used for training and 2 are used for validation.
    data_file = f"data/imagepatches_{train_args.patch_size}_seed{train_args.seed}.npz"
    if os.path.exists(data_file):
        data_file = np.load(data_file)
        data_patches, val_patches = data_file['data_patches'], data_file['val_patches']
        train_mean, train_std = data_file['train_mean'], data_file['train_std']
    else:
        data_patches = np.moveaxis(extract_patches_2d(images, (train_args.patch_size, train_args.patch_size)), -1, 1). \
                                   reshape(-1, train_args.patch_size, train_args.patch_size)
        np.random.shuffle(data_patches)
        
        train_mean, train_std = np.mean(data_patches, axis=0), np.std(data_patches, axis=0)
        val_idx = np.linspace(1, data_patches.shape[0] - 1, int(len(data_patches)*0.2), dtype=int)
        train_idx = np.ones(len(data_patches), bool)
        train_idx[val_idx] = 0
        val_patches = data_patches[val_idx]
        data_patches = data_patches[train_idx]
        
        np.savez_compressed(data_file, data_patches=data_patches, val_patches=val_patches, train_mean=train_mean, train_std=train_std)

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

    return train_patches, val_patches

def _calcx(theta,phi):
    x = np.multiply(np.sin(theta),np.cos(phi))
    y = np.multiply(np.sin(theta),np.sin(phi))
    z = np.cos(theta)
 
    return x,y,z

def create_spherical_data(batch_size,kappa=8):
    theta1 = np.random.rand(batch_size)*2*np.pi;
    phi1 = np.random.rand(batch_size)*2*np.pi;
    theta2 = np.random.vonmises(theta1,kappa)
    phi2 = np.random.vonmises(phi1,kappa)
    xtemp,ytemp,ztemp = _calcx(theta1,phi1)
    x0 = np.array([xtemp,ytemp,ztemp]).T
    xtemp,ytemp,ztemp = _calcx(theta2,phi2)
    x1 = np.array([xtemp,ytemp,ztemp]).T

    return x0, x1