import argparse
from tensorboardX import SummaryWriter
import glob
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torch import optim
import os.path
import h5py
import cnn
from function import adaptive_instance_normalization, coral

#import test

from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import datetime

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def open_file(filename):
    with h5py.File(filename, 'r') as f:
        # Get the data
        data = np.array(f['images'])
        labels = np.array(f['labels'])
    return data, labels

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def train_transform(img):
    transform_list = [
        transforms.RandomCrop(227),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)(img)

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=False,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=False,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

data_path = 'data'
optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

def batch_img(iterable, batch_size=32):
    iterable = list(iterable)
    l = len(iterable)
    for i in range(0, l, batch_size):
        yield torch.stack([content_tf(elem) for elem in iterable[i:min(i+batch_size, l)]])

def open_file(filename):
    with h5py.File(filename, 'r') as f:
        # Get the data
        data = np.array(f['images'])
        labels = np.array(f['labels'])
    return data, labels

def train(network, content_data, style_data, batch_size=32):
    all_count = correct_count = 0
    l = min(len(content_data), len(style_data))
    content_img_batches= batch_img(content_data, batch_size=batch_size)
    style_img_batches= batch_img(style_data, batch_size=batch_size)

    for i in range(l // batch_size):
        content_img = next(content_img_batches)
        content_img = content_img.to(device)
        style_img = next(style_img_batches)
        style_img = style_img.to(device)
        loss_c, loss_s = network(content_img, style_img)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i + 1))
            decoder_path = os.path.join(save_dir, 'decoder_iter_{:d}.pth.tar'.format(i + 1))



train_files = glob.glob('data/PACS/*train.hdf5')
target_domain = 'data/PACS/cartoon_train.hdf5'
style_data, _ = open_file(target_domain)

for i in range(1000):
        # shuffle data
    f = train_files[np.random.randint(0, len(train_files))]
    if f == target_domain:
        continue
    content_data, _ = open_file(f)
    print("Starting training on new domains...")
    print("Transferring from {} --> {}".format(target_domain, f))
    adjust_learning_rate(optimizer, iteration_count=i)
    train(network, content_data, style_data)
