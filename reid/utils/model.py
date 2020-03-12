
import torch.nn as nn
import copy


def init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_embedding(in_dim=None, out_dim=None, local_conv=False):
    layers = [
        nn.Conv2d(in_dim, out_dim, 1) if local_conv else nn.Linear(in_dim, out_dim),
        nn.BatchNorm2d(out_dim) if local_conv else nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    ]
    return nn.Sequential(*layers)

def get_temporal_features(temporal_features, t_f, epoch):
    if epoch == 0:
        temporal_features[0] = copy.deepcopy(t_f)
    elif epoch == 1:
        temporal_features[1] = copy.deepcopy(t_f)
    else:
        temporal_features[0] = copy.deepcopy(temporal_features[1])
        temporal_features[1] = copy.deepcopy(t_f)
    return temporal_features

