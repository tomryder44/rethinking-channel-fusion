import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

from trainers.trainer import Trainer


class MMDTrainer(Trainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)
        self.penalty = hyperparameters["penalty"]
        self.max_pairs = 50
        self.mmd_loss = SamplesLoss(loss="energy", backend="tensorized")

    def compute_mmd_loss(self, features, domains):
        dom_vals = torch.unique(domains)
        pairs = torch.combinations(dom_vals, r=2)
        if len(pairs) > self.max_pairs:
            perm = torch.randperm(len(pairs), device=features.device)
            pairs = pairs[perm[: self.max_pairs]]
        X_list, Y_list = [], []
        for d1, d2 in pairs:
            idx1 = (domains == d1).nonzero(as_tuple=True)[0]
            idx2 = (domains == d2).nonzero(as_tuple=True)[0]
            X_list.append(features.index_select(0, idx1))
            Y_list.append(features.index_select(0, idx2))
        X = torch.stack(X_list, dim=0)
        Y = torch.stack(Y_list, dim=0)
        pair_losses = self.mmd_loss(X, Y)
        return pair_losses.mean()

    def update(self, batch, training=True):
        x = batch["sequence"]
        y = batch["label"]
        d = batch["domain"]
        output = self.model(x)
        label_loss = F.cross_entropy(output["logits"], y)
        metrics = {"label_loss": label_loss.item()}
        if training is True:
            mmd_loss = self.compute_mmd_loss(output["features"], d)
            metrics["mmd"] = mmd_loss.item()
            total_loss = label_loss + self.penalty * mmd_loss
            self.optimiser.zero_grad()
            total_loss.backward()
            self.optimiser.step()
        return metrics
