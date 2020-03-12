from __future__ import print_function
import torch
from ..evaluation_metrics import accuracy

class IDLoss(object):
    def __init__(self, args):
        super(IDLoss, self).__init__()
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    def __call__(self, target, pred, step=0, **kwargs):
        # Calculation
        loss_list = [self.criterion(logits, target).mean() for logits in pred['logits_list']]
        prec, = accuracy(pred['logits_list'][1].data, target.data)
        loss = torch.stack(loss_list).sum()

        return {'loss': loss, 'prec': prec}
  
