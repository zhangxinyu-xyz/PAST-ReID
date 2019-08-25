from __future__ import absolute_import
import os
import sys

from .osutils import mkdir_if_missing
import datetime


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class ReDirectSTD(object):
    """Modified from Tong Xiao's `Logger` in open-reid.
    This class overwrites sys.stdout or sys.stderr, so that console logs can
    also be written to file.
    Args:
      fpath: file path
      console: one of ['stdout', 'stderr']
      immediately_visible: If `False`, the file is opened only once and closed
        after exiting. In this case, the message written to file may not be
        immediately visible (Because the file handle is occupied by the
        program?). If `True`, each writing operation of the console will
        open, write to, and close the file. If your program has tons of writing
        operations, the cost of opening and closing file may be obvious. (?)
    Usage example:
      `ReDirectSTD('stdout.txt', 'stdout', False)`
      `ReDirectSTD('stderr.txt', 'stderr', False)`
    NOTE: File will be deleted if already existing. Log dir and file is created
      lazily -- if no message is written, the dir and file will not be created.
    """

    def __init__(self, fpath=None, console='stdout', immediately_visible=False):
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == 'stdout' else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visible = immediately_visible
        if fpath is not None:
            # Remove existing log file.
            if os.path.exists(fpath):
                os.remove(fpath)

        # Overwrite
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            mkdir_if_missing(os.path.dirname(os.path.abspath(self.file)))
            if self.immediately_visible:
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')
                self.f.write(msg)
        self.flush()

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()

def array_str(array, fmt='{:.2f}', sep=', ', with_boundary=True):
    """String of a 1-D tuple, list, or numpy array containing digits."""
    ret = sep.join([fmt.format(float(x)) for x in array])
    if with_boundary:
        ret = '[' + ret + ']'
    return ret


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H-%M-%S'
    return datetime.datetime.today().strftime(fmt)
   
def print_log(stage, iteration, max_iters, epoch, batch_idx, num_dataloader, batch_time, data_time, losses, precisions, lr):
    
    if 'Conservative' in stage:
        print('Stage: {}\t'
              'Iteration: [{}/{}]\t'
              'Epoch: [{}][{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss {:.3f} ({:.3f})\t'
              'Lr {:.6f}\t'
              .format(stage,
                      iteration, max_iters,
                      epoch, batch_idx + 1, num_dataloader,
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      losses.val, losses.avg,
                      lr))
    elif 'Promoting' in stage:
        print('Stage: {}\t'
              'Iteration: [{}/{}]\t'
              'Epoch: [{}][{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss {:.3f} ({:.3f})\t'
              'Acc {:.3f} ({:.3f})\t'
              'Lr {:.6f}\t'
              .format(stage,
                      iteration, max_iters,
                      epoch, batch_idx + 1, num_dataloader,
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      losses.val, losses.avg,
                      precisions.val, precisions.avg,
                      lr))
    
    else:
        print('Please designate a training stage.')
        os._exit()



