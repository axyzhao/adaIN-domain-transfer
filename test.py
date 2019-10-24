import argparse
from pathlib import Path

import torch
import torch.nn as nn
import data_loaders
import cnn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
model = cnn.CNNet()
cnn = torch.load('cnn_classifier.pt')

vgg.to(device)
decoder.to(device)

data_path = 'data'

_, mnist_valid_loader = data_loaders.get_mnist_train_valid_loader(data_path,
                           1,
                           13090,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True)

_, svhn_valid_loader = data_loaders.get_mnist_train_valid_loader(data_path,
                           1,
                           13090,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True)

mnist_iter = iter(mnist_valid_loader)
svhn_iter = iter(svhn_valid_loader)

all_count = correct_count = 0
for mnist_img, mnist_label in mnist_iter:
    svhn_img, svhn_label = next(svhn_iter)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, svhn_img, mnist_img,
                                args.alpha)
        probabilities = model.forward(output).squeeze()
#    output = output.cpu()

    output_name = 'yar{}.png'.format(all_count) 
    #output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
    #    content_path.stem, style_path.stem, args.save_ext)

    true_label = svhn_label
    pred_label = probabilities.argmax()
    if (true_label == pred_label):
        correct_count += 1
    all_count += 1
    save_image(output, str(output_name))

print(correct_count / all_count)
