import argparse
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from pathlib import Path
from PIL import Image

# Model constants
CONTENT_LAYERS_DEFAULT = ['conv_4']
STYLE_LAYERS_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
STYLE_WEIGHT = 1e6
CONTENT_WEIGHT = 1e1
NUM_STEPS = 300
NUM_ITER = 4

# Training container paths
OUTPUT_DIR = '/opt/ml/output/data/'


class ContentLoss(nn.Module):
    """Module to compute style loss"""

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """Module to compute style loss based on Gram Matrix"""

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self._gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def _gram_matrix(self, input):
        a, b, c, d = input.size()
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class Normalization(nn.Module):
    """Module to normalize input image"""

    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]

    def __init__(self, mean=default_mean, std=default_std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        """Normalization step"""
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, style_img, content_img, device="cpu",
                               content_layers=CONTENT_LAYERS_DEFAULT,
                               style_layers=STYLE_LAYERS_DEFAULT):

    normalization = Normalization().to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a convolution
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(
    cnn, content_img, style_img, input_img,
    style_weight, content_weight,
    device="cpu", num_steps=NUM_STEPS,
    ):

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, device)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def model_loader(device="cpu"):
    return models.vgg19(pretrained=True).features.to(device).eval()


def image_loader(image_path: str, device: str):
    image = Image.open(image_path)
    loader = transforms.Compose([
        transforms.Resize(512),  # scale imported image
        transforms.ToTensor(),  # transform it into a torch tensor
    ])
    # Fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style_image_path = Path(args.style_data_dir) / args.style_image
    content_image_path = Path(args.content_data_dir) / args.content_image

    # Load images
    style_img = image_loader(style_image_path, device)
    content_img = image_loader(content_image_path, device)
    input_img = content_img.clone()

    # Resize style image
    target_shape = content_img.shape[2:]
    style_img_reshaped = transforms.Resize(target_shape)(style_img)

    cnn = model_loader(device)

    for i in range(args.num_iter):
        
        output = run_style_transfer(
            cnn, content_img, style_img_reshaped, input_img,
            style_weight=args.style_weight, content_weight=args.content_weight, 
            device=device, num_steps=args.num_steps)

        result_img = TF.to_pil_image(output.cpu()[0])
        result_img.save(Path(OUTPUT_DIR) / f'result-iter-{i}.jpg')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--content_image', type=str)
    parser.add_argument('--style_image', type=str)
    parser.add_argument('--style_weight', type=float, default=STYLE_WEIGHT)
    parser.add_argument('--content_weight', type=float, default=CONTENT_WEIGHT)
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS)
    parser.add_argument('--num_iter', type=int, default=NUM_ITER)
    parser.add_argument('--content_data_dir', type=str, default=os.environ['SM_CHANNEL_CONTENT_DATA']) # TODO: upload image to S3
    parser.add_argument('--style_data_dir', type=str, default=os.environ['SM_CHANNEL_STYLE_DATA']) # TODO: upload image to S3
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()

    print('='*80)
    print('Arguments passed')
    print('='*80)
    print(args)
    
    main(args)

    
