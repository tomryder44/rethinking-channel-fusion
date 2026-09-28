import torch

from utils.metrics import MetricComputer
from divergence.measure import measure_distances
import logging
logger = logging.getLogger(__name__)


def log_metrics(metrics, message=""):
    for key, value in metrics.items():
        message += f"{key} = {value:.2f}, "
    logger.debug(message)


def average_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}


class Trainer:

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        self.model = model.to(device)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.hyperparameters = hyperparameters
        self.device = device
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["lr"], weight_decay=self.hyperparameters["wd"])
        self.training_stats = {"train": {}, "val": {}, "test": {}}
        self.epoch = 0
        self.thr = 1e-3
        self.patience = 5
        self.metric_computer = MetricComputer()

    @torch.no_grad()
    def get_features(self, loader):
        self.model.eval()
        collected = None
        for batch in loader:
            x = batch["sequence"]
            output = self.model(x)
            step = {**output, "labels": batch["label"], "domains": batch["domain"]}
            if collected is None:
                collected = {k: [] for k in step}
            for k in step:
                collected[k].append(step[k])
        return {k: torch.cat(v, dim=0) for k, v in collected.items()}

    def update(self, batch, training=True):
        pass

    def train_one_epoch(self, source_train_loader):
        self.model.train()
        all_batch_metrics = []
        for batch in source_train_loader:
            batch_train_metrics = self.update(batch, training=True)
            all_batch_metrics.append(batch_train_metrics)
        metrics = average_metrics(all_batch_metrics)
        log_metrics(metrics, message="TRAIN STATS -- ")
        self.training_stats["train"][f"epoch {self.epoch}"] = metrics

    @torch.no_grad()
    def val_one_epoch(self, source_val_loader):
        self.model.eval()
        data = self.get_features(source_val_loader)
        metrics = self.metric_computer.compute(data)
        log_metrics(metrics, message="VAL STATS   -- ")
        self.training_stats["val"][f"epoch {self.epoch}"] = metrics

    @torch.no_grad()
    def test_one_epoch(self, target_loader):
        self.model.eval()
        data = self.get_features(target_loader)
        metrics = self.metric_computer.compute(data)
        log_metrics(metrics, message="TEST STATS   -- ")
        self.training_stats["test"][f"epoch {self.epoch}"] = metrics

    @torch.no_grad()
    def measure_domain_divergence(self, source_loader, target_loader):
        source_data = self.get_features(source_loader)
        target_data = self.get_features(target_loader)
        metrics = measure_distances(source_data["features"], source_data["domains"], target_data["features"])
        log_metrics(metrics, message="TEST STATS   -- ")
        self.training_stats["test"][f"epoch {self.epoch}"].update(metrics)

    def train(self, source_train_loader, source_val_loader, target_loader):
        self.target = torch.unique(target_loader.dataset.domain_labels).item()
        best_val_loss = float("inf")
        patience = self.patience
        for epoch in range(self.hyperparameters["num_epochs"]):
            self.epoch = epoch
            logger.debug(f"epoch: {self.epoch}/{self.hyperparameters['num_epochs']}")
            self.train_one_epoch(source_train_loader)
            self.val_one_epoch(source_val_loader)
            current_val_loss = self.training_stats["val"][f"epoch {self.epoch}"]["loss"]
            improvement = best_val_loss - current_val_loss
            if improvement > self.thr:
                best_val_loss = self.training_stats["val"][f"epoch {self.epoch}"]["loss"]
                patience = self.patience
            else:
                patience -= 1
            if patience <= 0:
                break
        self.test_one_epoch(target_loader)
        return self.training_stats
