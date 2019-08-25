from __future__ import absolute_import
import os

from PIL import Image

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            if len(indices) == 5:
                return self._get_triplet_item(indices)
            else:
                return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = os.path.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, index

    def _get_triplet_item(self, index):
        a_fname, pid1, _ = self.dataset[index[0]]
        p_fname, pid2, _ = self.dataset[index[2]]
        n_fname, pid3, _ = self.dataset[index[4]]

        a_img = Image.open(os.path.join(self.root, a_fname)).convert('RGB')
        p_img = Image.open(os.path.join(self.root, p_fname)).convert('RGB')
        n_img = Image.open(os.path.join(self.root, n_fname)).convert('RGB')
        if self.transform is not None:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        position = index[0:]

        pids = [pid1, pid2, pid3]
        return a_img, p_img, n_img, position, pids
