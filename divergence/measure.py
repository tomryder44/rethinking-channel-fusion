import torch

from divergence.mmd import compute_gaussian_mmd, compute_energy_mmd
from divergence.wasserstein import compute_wasserstein_distance


def get_weights(domains):
    unique_vals, counts = domains.unique(return_counts=True)
    weights = counts.float() / counts.sum()
    weight_dict = {int(k.item()): v.item() for k, v in zip(unique_vals, weights)}
    return weight_dict


def measure_weighted_sum(source_target, weights):
    ws = {key: source_target[key] * weights[key] for key in source_target}
    return sum(ws.values())


def measure_multi_source_target_dist(source_features, source_domains, target_features, distance_func):
    all_source_domains = torch.unique(source_domains)
    distances = {}
    for source_domain in all_source_domains:
        source_domain_features = source_features[source_domains == source_domain]
        distances[source_domain.item()] = distance_func(source_domain_features, target_features)
    return distances


def measure_multi_source_target_dist_gm(source_features, source_domains, target_features, bw):
    all_source_domains = torch.unique(source_domains)
    distances = {}
    for source_domain in all_source_domains:
        source_domain_features = source_features[source_domains == source_domain]
        distances[source_domain.item()] = compute_gaussian_mmd(source_domain_features, target_features, bw=bw)
    return distances


def median_pairwise(source_features, target_features):
    return torch.cdist(source_features, target_features, p=2).flatten().median().item()


def mean_pairwise(source_features, target_features):
    return torch.cdist(source_features, target_features, p=2).flatten().mean().item()


def centroid_distance(source_features, target_features):
    return (source_features.mean(dim=0) - target_features.mean(dim=0)).norm(p=2).item()


def measure_distances(source_features, source_domains, target_features):

    metrics = {}
    weights = get_weights(source_domains)

    # divergence metrics

    source_target_wasserstein = measure_multi_source_target_dist(source_features, source_domains, target_features, compute_wasserstein_distance)
    metrics["wasserstein"] = measure_weighted_sum(source_target_wasserstein, weights)

    source_target_mmd = measure_multi_source_target_dist(source_features, source_domains, target_features, compute_energy_mmd)
    metrics["emmd"] = measure_weighted_sum(source_target_mmd, weights)

    # for bw in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
    for bw in [0.25, 0.5, 1, 2, 4, 8, 16, 32]:
        d = measure_multi_source_target_dist_gm(source_features, source_domains, target_features, bw=bw)
        metrics[f"gmmd_{bw}"] = measure_weighted_sum(d, weights)

    # feature space metrics

    median_pw = measure_multi_source_target_dist(source_features, source_domains, target_features, median_pairwise)
    metrics["median_pairwise"] = measure_weighted_sum(median_pw, weights)

    mean_pw = measure_multi_source_target_dist(source_features, source_domains, target_features, mean_pairwise)
    metrics["mean_pairwise"] = measure_weighted_sum(mean_pw, weights)

    return metrics


# def measure_single_source_target_dist(source_features, target_features, distance_func):
#     return distance_func(source_features, target_features)
