import torch
import numpy as np

class CalClsParams():
    def __init__(self, name, dataset, features):
        self.dataset = dataset.train
        self.indices = dataset.train_indices
        self.features = features
        self.name = name

    def cal_cls_params(self):
        # feature-based weight initialization for classifier layer
        total_features = {}
        for input, indice in zip(self.dataset, self.indices):
            fname, label, _ = input
            feature = self.features[indice]
            if type(feature) is torch.Tensor:
                feature = feature.data.numpy()
            if label not in total_features.keys():
                total_features[label] = []
            total_features[label].append(feature)

        mean_features = []
        for label in sorted(total_features.keys()):
            mean_features.append(np.mean(total_features[label], axis=0))

        mean_features = torch.tensor(mean_features).cuda()
        return mean_features
