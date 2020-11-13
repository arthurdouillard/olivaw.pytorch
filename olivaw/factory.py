import torch
from torch import nn
from torch.nn import functional as F


def get_optimizer(parameters, args):
    if args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.grad_momentum,
            centered=True
        )
    elif args.optimizer == 'adam':
        return torch.optim.Adam(
            parameters,
            lr=args.lr
        )
    raise NotImplementedError(f'Unknown optimizer {args.optimizer}')


