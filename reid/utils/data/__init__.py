from __future__ import absolute_import

from .dataset import Dataset
from .preprocessor import Preprocessor
from .sampler import RandomIdentitySampler, SoftMarginTripletSampler, RandomIdSoftmarginTripletSampler

from torch.utils.data import DataLoader

from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
import os

__factory = {
    'CTL': RandomIdentitySampler,
    'RTL': SoftMarginTripletSampler,
    'CTL_RTL': RandomIdSoftmarginTripletSampler,
}


def names():
    return sorted(__factory.keys())


def create(name, data_source, *args, **kwargs):
    """
    Create a triplet sampler type.

    Parameters
    ----------
    name : str
        The triplet sampler type name. Can be one of 'randomidentity', 'softmargintriplet'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown triplet sampler type name:", name)
    return __factory[name](data_source, *args, **kwargs)

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

def create_train_data_loader(args, name, dataset, dist=None, istrain=False, idloss_only=False, savepath=None):
    train_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225])
    ])

    if args.triloss_use and istrain and (not idloss_only):
        sampler = create('CTL', dataset.train, args.batch_size, num_instances=args.num_instances, savepath=savepath)
        train_loader_CTL = DataLoader(
                Preprocessor(dataset.train, root=os.path.join(dataset.images_dir, dataset.train_path),
                             transform=train_transformer),
                batch_size=args.batch_size, num_workers=args.workers,
                sampler=sampler, pin_memory=True, drop_last=True)

        if args.tri_sampler_type == 'CTL_RTL':
            sampler = create(args.tri_sampler_type, dataset.train, dist=dist, k=args.k_nearest, data_indices=dataset.train_indices, savepath=savepath)
            train_loader_RTL = DataLoader(
                Preprocessor(dataset.train, root=os.path.join(dataset.images_dir, dataset.train_path),
                             transform=train_transformer),
                batch_size=args.batch_size, num_workers=args.workers,
                sampler=sampler, pin_memory=True, drop_last=True)
            
            print('{} Datasets Has beed loaded.'.format(name))
            return train_loader_CTL, train_loader_RTL

    elif istrain:
        if idloss_only:
            train_loader = DataLoader(
                Preprocessor(dataset.train, root=os.path.join(dataset.images_dir, dataset.train_path),
                             transform=train_transformer),
                batch_size=args.batch_size, num_workers=args.workers,
                shuffle=True, pin_memory=True, drop_last=True)
        else:
            assert 0, 'There is no train dataloader.'
    else:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=os.path.join(dataset.images_dir, dataset.train_path),
                         transform=train_transformer),
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False, pin_memory=True, drop_last=False)

    print('{} Datasets Has beed loaded.'.format(name))

    return train_loader, None

