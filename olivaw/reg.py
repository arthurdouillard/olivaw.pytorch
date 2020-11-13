import torch
from torch import nn
from torch.nn import functional as F


def srank_penalty(features_mat, factor=0.001):
    """Penality avoiding srank collapse.

    # Reference:
        * Implicit Under-Parametrization Inhibits Data-Efficient Deep Reinforcement Learning
          Kumar et al. 2020

    :param features_mat: The features matrix before the last FC layer.
    :param factor: A factor weighting the importance of the penality.
    :return: A regularization loss to backward on.
    """
    s = torch.svd(features_mat, some=False, compute_uv=False)[1]

    s_max = s.max() ** 2
    s_min = s.min() ** 2

    return factor * (s_max - s_min)
