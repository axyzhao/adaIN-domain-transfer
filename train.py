import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torch import optim
import os.path
import data_loaders
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


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


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

device = torch.device('cpu')
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


"""content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
"""
optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    if i % 6000 == 0:
        mnist_train_loader, mnist_valid_loader = data_loaders.get_mnist_train_valid_loader(data_path,
                           args.batch_size,
                           13090,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True)
        svhn_train_loader, svhn_valid_loader = data_loaders.get_mnist_train_valid_loader(data_path,
                           args.batch_size,
                           13090,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True)

        style_iter = iter(mnist_train_loader)
        content_iter = iter(svhn_train_loader)

    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = torch.tensor(next(content_iter)[0])#.to(device)
    style_images = torch.tensor(next(style_iter)[0])#.to(device)

    loss_c, loss_s = network(content_images, style_images)
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
    #    for key in state_dict.keys():
    #        state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))
        decoder_path = os.path.join(save_dir, 'decoder_iter_{:d}.pth.tar'.format(i + 1))

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(decoder_path))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

mnist_train_loader, mnist_valid_loader = data_loaders.get_mnist_train_valid_loader(data_path,
                           1,
                           13090,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True)

svhn_train_loader, svhn_valid_loader = data_loaders.get_mnist_train_valid_loader(data_path,
                           1,
                           13090,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True)

mnist_iter = iter(mnist_train_loader)
svhn_iter = iter(svhn_train_loader)
model = cnn.CNNet()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
show_every = 100

for i in tqdm(range(args.max_iter)):
    if i % 6000 == 0:
        mnist_train_loader, mnist_valid_loader = data_loaders.get_mnist_train_valid_loader(data_path,
                           1,
                           13090,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True)
        svhn_train_loader, svhn_valid_loader = data_loaders.get_mnist_train_valid_loader(data_path,
                           1,
                           13090,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True)

        mnist_iter = iter(mnist_train_loader)
        svhn_iter = iter(svhn_train_loader)

#    content_tf = test_transform(args.content_size, args.crop)
#    style_tf = test_transform(args.style_size, args.crop)
    mnist_img, mnist_label = next(mnist_iter)
    svhn_img, svhn_label = next(svhn_iter)
    optimizer.zero_grad()

    if np.random.binomial(1, 0.5):
        output = style_transfer(vgg, decoder, svhn_img, mnist_img,
                            args.alpha)
    else:
        output = svhn_img
    probabilities = model.forward(output)#.squeeze()
    loss = model.loss(probabilities, svhn_label)
    if i % show_every == 0:
      #  imshow(utils.make_grid(mnist))
        print("Time: {}".format(datetime.datetime.now()))
        print("Loss at step {} is {}".format(i, loss))
    loss.backward()
    optimizer.step()

torch.save(model, 'cnn_classifier.pt')
writer.close()
