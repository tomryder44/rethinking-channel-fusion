import torch
import torch.nn.functional as F

from trainers.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


class DiversifyTrainer(Trainer):

    def __init__(self, model, num_classes, num_channels, device, hyperparameters):
        super().__init__(model, num_classes, num_channels, device, hyperparameters)
        self.K = int(hyperparameters["K"])
        self.alpha1 = 1.0
        self.alpha = 1.0
        self.lam = 0.0
        self.local_epochs = 1
        self.num_classes = num_classes
        self.pseudo_domains = None
        self.opt_adv = None
        self.opt_cls = None
        self.opt_all = None

    def _build_optimisers(self):
        lr = self.hyperparameters["lr"]
        wd = self.hyperparameters["wd"]
        m = self.model
        self.opt_adv = torch.optim.Adam(
            list(m.dbottleneck.parameters()) + list(m.dclassifier.parameters())
            + list(m.ddiscriminator.parameters()), lr=lr, weight_decay=wd)
        self.opt_cls = torch.optim.Adam(
            list(m.bottleneck.parameters()) + list(m.classifier.parameters())
            + list(m.discriminator.parameters()), lr=lr, weight_decay=wd)
        self.opt_all = torch.optim.Adam(
            list(m.feature_extractor.parameters()) + list(m.abottleneck.parameters())
            + list(m.aclassifier.parameters()), lr=lr, weight_decay=wd)

    def _materialise_and_build(self, loader):
        batch = next(iter(loader))
        x = batch["sequence"].to(self.device)
        self.model.train()
        with torch.no_grad():
            self.model.featurize(x)
            self.model.d_featurize(x)
            self.model.a_featurize(x)
        self._build_optimisers()

    def _entropy(self, logits):
        p = F.softmax(logits, dim=1)
        return -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def _iter_batches(self, loader):
        for batch in loader:
            yield (batch["index"].to(self.device),
                   batch["sequence"].to(self.device),
                   batch["label"].to(self.device))

    def _init_pseudo_domains(self, loader):
        n = len(loader.dataset)
        self.pseudo_domains = torch.zeros(n, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update_pseudo_domains(self, loader):
        self.model.eval()
        all_z, all_soft, all_idx = [], [], []
        for idx, x, _ in self._iter_batches(loader):
            z = self.model.d_featurize(x)
            soft = F.softmax(self.model.d_class_logits(z), dim=1)
            all_z.append(z)
            all_soft.append(soft)
            all_idx.append(idx)
        z = torch.cat(all_z, dim=0)
        soft = torch.cat(all_soft, dim=0)
        idx = torch.cat(all_idx, dim=0)

        z_aug = torch.cat([z, torch.ones(z.size(0), 1, device=z.device)], dim=1)
        z_aug = z_aug / z_aug.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        z_norm = F.normalize(z_aug, dim=1)

        assign = None
        for r in range(2):
            aff = soft if r == 0 else F.one_hot(assign, self.K).float()
            centroids = aff.t() @ z_aug
            centroids = centroids / aff.sum(dim=0).clamp_min(1e-8).unsqueeze(1)
            dist = 1 - z_norm @ F.normalize(centroids, dim=1).t()
            assign = dist.argmin(dim=1)
        self.pseudo_domains[idx] = assign

    def step_fine_grained(self, loader):
        self.model.train()
        for _ in range(self.local_epochs):
            for idx, x, y in self._iter_batches(loader):
                if x.size(0) == 1:
                    continue
                d_prime = self.pseudo_domains[idx]
                s = d_prime * self.num_classes + y
                z = self.model.a_featurize(x)
                loss = F.cross_entropy(self.model.super_classify(z), s)
                self.opt_all.zero_grad()
                loss.backward()
                self.opt_all.step()

    def step_characterize(self, loader):
        self.model.train()
        for _ in range(self.local_epochs):
            for idx, x, y in self._iter_batches(loader):
                if x.size(0) == 1:
                    continue
                d_prime = self.pseudo_domains[idx]
                z = self.model.d_featurize(x)
                adv_logits = self.model.d_discriminate(z, self.alpha1)
                disc_loss = F.cross_entropy(adv_logits, y)
                dom_logits = self.model.d_class_logits(z)
                ent_loss = self.lam * self._entropy(dom_logits) + F.cross_entropy(dom_logits, d_prime)
                loss = disc_loss + ent_loss
                self.opt_adv.zero_grad()
                loss.backward()
                self.opt_adv.step()

    def step_invariant(self, loader):
        self.model.train()
        for _ in range(self.local_epochs):
            for idx, x, y in self._iter_batches(loader):
                if x.size(0) == 1:
                    continue
                d_prime = self.pseudo_domains[idx]
                z = self.model.featurize(x)
                cls_loss = F.cross_entropy(self.model.classify(z), y)
                dom_logits = self.model.domain_classify(z, self.alpha)
                dom_loss = F.cross_entropy(dom_logits, d_prime)
                loss = cls_loss + dom_loss
                self.opt_cls.zero_grad()
                loss.backward()
                self.opt_cls.step()

    def train(self, source_train_loader, source_val_loader, target_loader):
        self._materialise_and_build(source_train_loader)
        self.target = torch.unique(target_loader.dataset.domain_labels).item()
        self._init_pseudo_domains(source_train_loader)
        best_val_loss = float("inf")
        patience = self.patience
        for epoch in range(self.hyperparameters["num_epochs"]):
            self.epoch = epoch
            logger.debug(f"epoch: {epoch}/{self.hyperparameters['num_epochs']}")
            self.step_fine_grained(source_train_loader)
            self.update_pseudo_domains(source_train_loader)
            self.step_characterize(source_train_loader)
            self.step_invariant(source_train_loader)
            self.training_stats["train"][f"epoch {self.epoch}"] = {"loss": 0.0}
            self.val_one_epoch(source_val_loader)
            current_val_loss = self.training_stats["val"][f"epoch {self.epoch}"]["loss"]
            improvement = best_val_loss - current_val_loss
            if improvement > self.thr:
                best_val_loss = current_val_loss
                patience = self.patience
            else:
                patience -= 1
            if patience <= 0:
                break
        self.test_one_epoch(target_loader)
        return self.training_stats