import math

import torch
import torch.nn.functional as F

from trainers.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


class GroupDROTrainer(Trainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)
        self.eta = hyperparameters["penalty"]
        self.group_weights = {}

    def compute_dro_loss(self, logits, labels, domains):
        dom_ids = torch.unique(domains).tolist()
        dom_losses = {}
        for dom in dom_ids:
            idx = torch.where(domains == dom)[0]
            dom_losses[dom] = F.cross_entropy(logits[idx], labels[idx])
            if dom not in self.group_weights:
                self.group_weights[dom] = 1.0
        for dom in dom_ids:
            self.group_weights[dom] *= math.exp(self.eta * dom_losses[dom].item())
        total = sum(self.group_weights.values())
        for dom in self.group_weights:
            self.group_weights[dom] /= total
        dro_loss = sum(self.group_weights[dom] * dom_losses[dom] for dom in dom_ids)
        return dro_loss

    def update(self, batch, training=True):
        x = batch["sequence"]
        y = batch["label"]
        d = batch["domain"]
        output = self.model(x)
        label_loss = F.cross_entropy(output["logits"], y)
        metrics = {"label_loss": label_loss.item()}
        if training:
            dro_loss = self.compute_dro_loss(output["logits"], y, d)
            metrics["dro"] = dro_loss.item()
            metrics["max_group_weight"] = max(self.group_weights.values())
            self.optimiser.zero_grad()
            dro_loss.backward()
            self.optimiser.step()
        return metrics