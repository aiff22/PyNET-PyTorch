# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torchvision import models
import torch.nn as nn
import torch

CONTENT_LAYER = 'relu_16'


def vgg_19(device):

    vgg_19 = models.vgg19(pretrained=True).features
    model = nn.Sequential()

    i = 0
    for layer in vgg_19.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name == CONTENT_LAYER:
            break

    model = model.to(device)
    model = torch.nn.DataParallel(model)

    for param in model.parameters():
        param.requires_grad = False

    for param in vgg_19.parameters():
        param.requires_grad = False

    return model
