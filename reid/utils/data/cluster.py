import numpy as np
import copy
import hdbscan
from sklearn.cluster import DBSCAN
import os

class Cluster(object):
    def __init__(self, args, traindataset, tblogger=None):
        self.args = args
        self.traindataset = traindataset
        self.old_indices = range(len(self.traindataset))
        # self.tblogger = tblogger

    def hdbscancluster(self, dist, iteration=-1):
        # HDBSCAN cluster
        clusterer = hdbscan.HDBSCAN(min_samples=self.args.dbscan_minsample, metric='precomputed')  # min_cluster_size=2,
        labels = clusterer.fit_predict(dist.astype(np.double))

        # select & cluster images as training set of this iteration
        print('Clustering and labeling...')
        num_ids = len(set(labels)) - 1

        print('Iteration {} have {} training ids'.format(iteration, num_ids))
        # generate new dataset
        new_dataset = []
        new_indices = []

        for (fname, _, _), label, indice in zip(self.traindataset, labels, self.old_indices):
            if label == -1:
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname, label, indice))
            new_indices.append(indice)

        print('Iteration {} have {} training images'.format(iteration, len(new_dataset)))

        return new_dataset, new_indices

    def dbscancluster(self, dist, iteration=-1):
        # DBSCAN cluster
        tri_mat = np.triu(dist, 1)  # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
        tri_mat = np.sort(tri_mat, axis=None)
        top_num = np.round(self.args.rho * tri_mat.size).astype(int)
        eps = tri_mat[:top_num].mean()
        print('eps in cluster: {:.3f}'.format(eps))

        clusterer = DBSCAN(eps=eps, min_samples=self.args.dbscan_minsample, metric='precomputed', n_jobs=8)

        labels = clusterer.fit_predict(dist)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        num_ids = len(set(labels)) - 1

        print('Epoch {} have {} training ids'.format(iteration, num_ids))
        # generate new dataset
        new_dataset = []
        new_indices = []

        for (fname, _, _), label, indice in zip(self.traindataset, labels, self.old_indices):
            if label == -1:
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname, label, indice))
            new_indices.append(indice)

        print('Iteration {} have {} training images'.format(iteration, len(new_dataset)))

        return new_dataset, new_indices
