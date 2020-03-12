import torch.optim as optim


def create_optimizer(param_groups, args):
    if args.optimizer == 'sgd':
        optim_class = optim.SGD
    elif args.optimizer == 'adam':
        optim_class = optim.Adam
    else:
        raise NotImplementedError('Unsupported Optimizer {}'.format(args.optimizer))
    optim_kwargs = dict(weight_decay=args.weight_decay)
    if args.optimizer == 'sgd':
        optim_kwargs['momentum'] = args.momentum
        optim_kwargs['nesterov'] = args.nesterov
    return optim_class(param_groups, **optim_kwargs)
