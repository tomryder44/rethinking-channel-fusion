import torch
import torch.nn.functional as F

from trainers.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


class VREXTrainer(Trainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)
        self.penalty = hyperparameters["penalty"]

    def compute_vrex_loss(self, logits, labels, domains):
        dom_ids = torch.unique(domains).tolist()
        losses = []
        for dom in dom_ids:
            idx = torch.where(domains == dom)[0]
            if idx.numel() == 0:
                continue
            losses.append(F.cross_entropy(logits[idx], labels[idx]))
        losses = torch.stack(losses)
        var_loss = losses.var(unbiased=False)
        return var_loss

    def update(self, batch, training=True):
        x = batch["sequence"]
        y = batch["label"]
        d = batch["domain"]
        output = self.model(x)
        label_loss = F.cross_entropy(output["logits"], y)
        metrics = {"label_loss": label_loss.item()}
        if training:
            vrex_loss = self.compute_vrex_loss(output["logits"], y, d)
            metrics["vrex"] = vrex_loss.item()
            total_loss = label_loss + self.penalty * vrex_loss
            self.optimiser.zero_grad()
            total_loss.backward()
            self.optimiser.step()
        return metrics
