import torch
import ot


def compute_wasserstein_distance(source_features, target_features, metric="euclidean"):
    n, m = source_features.shape[0], target_features.shape[0]
    a = torch.ones(n) / n
    b = torch.ones(m) / m
    a = a.to(source_features.device)
    b = b.to(source_features.device)
    M = ot.dist(source_features, target_features, metric=metric)
    wasserstein_dist = ot.emd2(a, b, M, numItermax=1000000)
    return wasserstein_dist.item()
