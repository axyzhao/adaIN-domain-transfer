from torch import nn
import numpy as np
import torch
from torch import nn, optim

from collections import OrderedDict
from torchvision import datasets, transforms
from torchvision import utils
from torch.utils.data.sampler import SubsetRandomSampler

def get_mnist_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms

    transform = transforms.Compose([
                    transforms.Pad(2),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load the dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                download=True, transform=transform)

    valid_dataset = datasets.MNIST(root=data_dir, train=True,
                download=True, transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=1, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)


    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=9,
                                                    shuffle=shuffle,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        plot_images(X, labels)

    return (train_loader, valid_loader)


    def get_svhn_train_valid_loader(data_dir,
                               batch_size,
                               random_seed,
                               valid_size=0.2,
                               shuffle=True,
                               show_sample=False,
                               num_workers=1,
                               pin_memory=True):
        """
        Utility function for loading and returning train and valid
        multi-process iterators over the MNIST dataset. A sample
        9x9 grid of the images can be optionally displayed.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - augment: whether to apply the data augmentation scheme
          mentioned in the paper. Only applied on the train split.
        - random_seed: fix seed for reproducibility.
        - valid_size: percentage split of the training set used for
          the validation set. Should be a float in the range [0, 1].
        - shuffle: whether to shuffle the train/validation indices.
        - show_sample: plot 9x9 sample grid of the dataset.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
          True if using GPU.
        Returns
        -------
        - train_loader: training set iterator.
        - valid_loader: validation set iterator.
        """
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

        # define transforms

        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load the dataset
        train_dataset = datasets.SVHN(root=data_dir, split='train',
                    download=True, transform=transform)

        valid_dataset = datasets.SVHN(root=data_dir, split='test',
                    download=True, transform=transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle == True:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, sampler=train_sampler,
                        num_workers=num_workers, pin_memory=pin_memory)

        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                        batch_size=1, sampler=valid_sampler,
                        num_workers=num_workers, pin_memory=pin_memory)


        # visualize some images
        if show_sample:
            sample_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=9,
                                                        shuffle=shuffle,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)
            data_iter = iter(sample_loader)
            images, labels = data_iter.next()
            X = images.numpy()
            plot_images(X, labels)

        return (train_loader, valid_loader)
