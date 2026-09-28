import torch
import torch.nn.functional as F

from trainers.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


class NNRTrainer(Trainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)
        self.penalty = hyperparameters["penalty"]

    def update(self, batch, training=True):
        x = batch["sequence"]
        y = batch["label"]
        output = self.model(x)
        label_loss = F.cross_entropy(output["logits"], y)
        metrics = {"label_loss": label_loss.item()}
        if training:
            features = output["features"]
            _, s, _ = torch.linalg.svd(features)
            nnr_loss = torch.sum(s)
            metrics["nnr"] = nnr_loss.item()
            total_loss = label_loss + self.penalty * nnr_loss
            self.optimiser.zero_grad()
            total_loss.backward()
            self.optimiser.step()
        return metrics
