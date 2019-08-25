from __future__ import absolute_import

from .resnet import get_resnet

__factory = {
    'resnet18': get_resnet,
    'resnet34': get_resnet,
    'resnet50': get_resnet,
    'resnet101': get_resnet,
    'resnet152': get_resnet,
}


def names():
    return sorted(__factory.keys())


def create_backbone(name, args):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](args)
