from __future__ import absolute_import
import os
import errno
import scipy.io as scio


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_mat(savepath, filename, matname, mat):
    mkdir_if_missing(savepath)
    scio.savemat(os.path.join(savepath, filename), {matname: mat})

def load_mat(savepath, datasetname, filename, matname):
    if savepath == filename:
        mat = scio.loadmat(savepath)[matname]
    else:
        mat = scio.loadmat(os.path.join(savepath, datasetname, filename))[matname]
    return mat


