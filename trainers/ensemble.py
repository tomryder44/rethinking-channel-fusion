import torch

from trainers.trainer import Trainer
from utils.metrics import EnsembleMetricComputer


class EnsembleTrainer(Trainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)
        self.metric_computer = EnsembleMetricComputer()

    @torch.no_grad()
    def get_features(self, loader, weights=None):
        if weights is None:
            return super().get_features(loader)
        self.model.eval()
        collected = None
        for batch in loader:
            x = batch["sequence"]
            output = self.model(x, weights)
            step = {**output, "labels": batch["label"], "domains": batch["domain"]}
            if collected is None:
                collected = {k: [] for k in step}
            for k in step:
                collected[k].append(step[k])
        return {k: torch.cat(v, dim=0) for k, v in collected.items()}
