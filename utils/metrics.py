import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from divergence.measure import measure_distances
import logging
logger = logging.getLogger(__name__)


class MetricComputer:

    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels).item()

    def compute_macro_f1(self, preds, labels):
        preds = torch.argmax(preds, dim=1)
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        score = f1_score(labels, preds, average="macro")
        return float(score)

    def compute(self, data):
        logits = data["logits"]
        labels = data["labels"]
        metrics = {}
        metrics["loss"] = self.compute_loss(logits, labels)
        metrics["macro_f1"] = self.compute_macro_f1(logits, labels)
        return metrics


class EnsembleMetricComputer(MetricComputer):

    def compute_loss_pr(self, probs, labels):
        return F.nll_loss(torch.log(probs + 1e-9), labels).item()

    def compute(self, data):
        probs = data["probs"]
        labels = data["labels"]
        metrics = {}
        metrics["loss"] = self.compute_loss_pr(probs, labels)
        metrics["macro_f1"] = self.compute_macro_f1(probs, labels)
        all_logits = data["all_logits"]
        for m in range(all_logits.shape[1]):  # num models in the ensemble
            m_logits = all_logits[:, m]
            metrics[f"{m}_loss"] = self.compute_loss(m_logits, labels)
            metrics[f"{m}_macro_f1"] = self.compute_macro_f1(m_logits, labels)
        return metrics
