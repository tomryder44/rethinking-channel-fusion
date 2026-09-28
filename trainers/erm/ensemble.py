import torch
import torch.nn.functional as F

from trainers.ensemble import EnsembleTrainer


class EnsembleERMTrainer(EnsembleTrainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)

    def update(self, batch, training=True):
        x = batch["sequence"]
        y = batch["label"]
        out = self.model(x)
        total_label_loss = 0.
        metrics = {}
        num_models = out["all_logits"].shape[1]
        for i in range(num_models):
            label_loss_i = F.cross_entropy(out["all_logits"][:, i], y)
            total_label_loss += label_loss_i
        if training is True:
            self.optimiser.zero_grad()
            total_label_loss.backward()
            self.optimiser.step()
        combined_loss = F.nll_loss(torch.log(out["probs"] + 1e-9), y).item()
        metrics["loss"] = combined_loss
        return metrics
