import torch.nn.functional as F

from trainers.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


class EarlyERMTrainer(Trainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)

    def update(self, batch, training=True):
        x = batch["sequence"]
        y = batch["label"]
        output = self.model(x)
        label_loss = F.cross_entropy(output["logits"], y)
        if training:
            self.optimiser.zero_grad()
            label_loss.backward()
            self.optimiser.step()
        metrics = {"loss": label_loss.item()}
        return metrics
