import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from paths import PROCESSED_DIR
from utils.torch_datasets import TimeSeriesDataset, STFTDataset
from utils.torch_loaders import DomainSampler
from utils.aug import compute_hilbert_transform

import logging
logger = logging.getLogger(__name__)


def stratified_split(dataset: tuple, train_size: float):
    data = dataset[0]
    targets = dataset[1]
    test_size = 1 - train_size
    train_x, test_x, train_y, test_y = train_test_split(
        data, targets, test_size=test_size, stratify=targets)
    return train_x, train_y, test_x, test_y


def normalise(x):
    mu = np.mean(x, axis=(0, 2), keepdims=True)
    sigma = np.std(x, axis=(0, 2), keepdims=True)
    x = (x - mu) / (sigma + 1e-8)
    return x


def load_domain(dataset_path: str, domain: int) -> tuple:
    x = np.load(os.path.join(dataset_path, f"{domain}_x.npy"))
    y = np.load(os.path.join(dataset_path, f"{domain}_y.npy"))
    values, counts = np.unique(y, return_counts=True)
    mask = np.isin(y, values[counts >= 2])
    return normalise(x[mask]), y[mask]


def load_arrays(dataset, source, target, train_size=0.75):

    dataset_path = os.path.join(PROCESSED_DIR, dataset)

    source_train_x, source_val_x = [], []
    source_train_y, source_val_y = [], []
    source_train_domains, source_val_domains = [], []
    for source_domain in source:
        source_x, source_y = load_domain(dataset_path, source_domain)
        train_x, train_y, val_x, val_y = stratified_split((source_x, source_y), train_size=train_size)
        source_train_x.append(train_x)
        source_val_x.append(val_x)
        source_train_y.append(train_y)
        source_val_y.append(val_y)
        source_train_domains.append(np.full(len(train_x), int(source_domain)))
        source_val_domains.append(np.full(len(val_x), int(source_domain)))
    source_train_x = np.concatenate(source_train_x)
    source_val_x = np.concatenate(source_val_x)
    source_train_y = np.concatenate(source_train_y)
    source_val_y = np.concatenate(source_val_y)
    source_train_domains = np.concatenate(source_train_domains)
    source_val_domains = np.concatenate(source_val_domains)

    targets_x = []
    targets_y = []
    target_domains = []
    for target_domain in target:
        target_x, target_y = load_domain(dataset_path, target_domain)
        targets_x.append(target_x)
        targets_y.append(target_y)
        target_domains.append(np.full(len(target_x), int(target_domain)))
    targets_x = np.concatenate(targets_x)
    targets_y = np.concatenate(targets_y)
    target_domains = np.concatenate(target_domains)

    # keep common classes only
    source_train_cls = np.unique(source_train_y)
    source_val_cls = np.unique(source_val_y)
    target_cls = np.unique(targets_y)
    common_cls = np.intersect1d(np.intersect1d(source_train_cls, source_val_cls), target_cls)
    source_train_mask = np.isin(source_train_y, common_cls)
    source_val_mask = np.isin(source_val_y, common_cls)
    target_mask = np.isin(targets_y, common_cls)
    source_train_x = source_train_x[source_train_mask]
    source_train_y = source_train_y[source_train_mask]
    source_train_domains = source_train_domains[source_train_mask]
    source_val_x = source_val_x[source_val_mask]
    source_val_y = source_val_y[source_val_mask]
    source_val_domains = source_val_domains[source_val_mask]
    targets_x = targets_x[target_mask]
    targets_y = targets_y[target_mask]
    target_domains = target_domains[target_mask]

    cls_to_new_label = {cls: i for i, cls in enumerate(common_cls)}
    source_train_y = np.array([cls_to_new_label[y] for y in source_train_y])
    source_val_y = np.array([cls_to_new_label[y] for y in source_val_y])
    targets_y = np.array([cls_to_new_label[y] for y in targets_y])

    return (source_train_x, source_train_y, source_train_domains), \
           (source_val_x, source_val_y, source_val_domains), \
           (targets_x, targets_y, target_domains)


def load_domain_sampling(dataset, source, target, device, batch_size, train_size):
    train, val, tgt = load_arrays(dataset, source, target, train_size)
    source_train = TimeSeriesDataset(*train, device)
    source_val = TimeSeriesDataset(*val, device)
    target_train = TimeSeriesDataset(*tgt, device)
    train_sampler = DomainSampler(source_train, samples_per_domain=16)
    source_train_loader = DataLoader(source_train, batch_sampler=train_sampler)
    source_val_loader = DataLoader(source_val, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_train, batch_size=batch_size, shuffle=False)
    return source_train_loader, source_val_loader, target_loader


def load_phaser_sampling(dataset, source, target, device, batch_size, train_size, nperseg_k=None):
    train, val, tgt = load_arrays(dataset, source, target, train_size)
    train = compute_hilbert_transform(*train)
    source_train = STFTDataset(*train, device, nperseg_k)
    source_val = STFTDataset(*val, device, nperseg_k)
    target_train = STFTDataset(*tgt, device, nperseg_k)
    source_train_loader = DataLoader(source_train, batch_size=batch_size, shuffle=True)
    source_val_loader = DataLoader(source_val, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_train, batch_size=batch_size, shuffle=False)
    return source_train_loader, source_val_loader, target_loader


def load_random_sampling(dataset, source, target, device, batch_size, train_size):
    train, val, tgt = load_arrays(dataset, source, target, train_size)
    source_train = TimeSeriesDataset(*train, device)
    source_val = TimeSeriesDataset(*val, device)
    target_train = TimeSeriesDataset(*tgt, device)
    source_train_loader = DataLoader(source_train, batch_size=batch_size, shuffle=True)
    source_val_loader = DataLoader(source_val, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_train, batch_size=batch_size, shuffle=False)
    return source_train_loader, source_val_loader, target_loader


def load_dataloaders(dataset, source, target, device, algorithm="erm",
                     batch_size=64, train_size=0.75, nperseg_k=None):

    if algorithm in ["mmd", "vrex", "groupdro", "irm"]:
        return load_domain_sampling(dataset, source, target, device, batch_size, train_size)

    elif algorithm == "phaser":
        return load_phaser_sampling(dataset, source, target, device, batch_size, train_size, nperseg_k)

    else:
        return load_random_sampling(dataset, source, target, device, batch_size, train_size)
