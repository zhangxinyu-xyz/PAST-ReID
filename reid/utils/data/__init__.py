from __future__ import absolute_import

from .dataset import Dataset
from .preprocessor import Preprocessor

from torch.utils.data import DataLoader
from reid.utils.data import transforms as T
import os

def create_test_data_loader(args, name, dataset):
    train_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        T.Normalize(mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225])
    ])

    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        T.Normalize(mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=os.path.join(dataset.images_dir, dataset.train_path),
                     transform=train_transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True, drop_last=False)

    query_loader = DataLoader(
        Preprocessor(dataset.query, root=os.path.join(dataset.images_dir, dataset.query_path),
                     transform=test_transformer),
        batch_size=args.batch_size*4, num_workers=args.workers,
        shuffle=False, pin_memory=True, drop_last=False)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=os.path.join(dataset.images_dir, dataset.gallery_path),
                     transform=test_transformer),
        batch_size=args.batch_size*4, num_workers=args.workers,
        shuffle=False, pin_memory=True, drop_last=False)

    print('{} Datasets Has beed loaded.'.format(name))

    return train_loader, query_loader, gallery_loader


