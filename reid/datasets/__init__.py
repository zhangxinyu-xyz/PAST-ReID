from __future__ import absolute_import
from .duke import Duke
from .market1501 import Market1501
from .cuhk03 import CUHK03

__factory = {
    'market1501': Market1501,
    'duke': Duke,
    'cuhk03': CUHK03
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'market', 'duke'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)



