import torch
import torch.autograd as autograd
import torch.nn.functional as F

from trainers.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


class IRMTrainer(Trainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)
        self.penalty = hyperparameters["penalty"]

    def compute_irm_penalty(self, logits, labels, domains):
        dom_ids = torch.unique(domains).tolist()
        penalties = []
        for dom in dom_ids:
            idx = torch.where(domains == dom)[0]
            if idx.numel() < 2:
                continue
            dom_logits = logits[idx]
            dom_labels = labels[idx]
            scale = torch.ones(1, device=logits.device, requires_grad=True)
            loss_a = F.cross_entropy(dom_logits[::2] * scale, dom_labels[::2])
            loss_b = F.cross_entropy(dom_logits[1::2] * scale, dom_labels[1::2])
            grad_a = autograd.grad(loss_a, [scale], create_graph=True)[0]
            grad_b = autograd.grad(loss_b, [scale], create_graph=True)[0]
            penalties.append(torch.sum(grad_a * grad_b))
        if len(penalties) == 0:
            return logits.new_zeros(())
        return torch.stack(penalties).mean()

    def update(self, batch, training=True):
        x = batch["sequence"]
        y = batch["label"]
        d = batch["domain"]
        output = self.model(x)
        label_loss = F.cross_entropy(output["logits"], y)
        metrics = {"label_loss": label_loss.item()}
        if training:
            irm_penalty = self.compute_irm_penalty(output["logits"], y, d)
            metrics["irm"] = irm_penalty.item()
            total_loss = label_loss + self.penalty * irm_penalty
            self.optimiser.zero_grad()
            total_loss.backward()
            self.optimiser.step()
        return metrics