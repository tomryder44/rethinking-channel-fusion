import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from train import set_seed
from utils.split import get_loso_splits
from utils.load import load_dataloaders

from trainers.erm.early import EarlyERMTrainer


class CNNEncoder(nn.Module):

    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=num_filters, kernel_size=3, bias=False)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        a1 = self.relu(self.bn1(self.conv1(x)))
        a2 = self.relu(self.bn2(self.conv2(a1)))
        a3 = self.relu(self.bn3(self.conv3(a2)))
        out = a3.mean(dim=2)
        return out, [a1, a2, a3]


class EarlyFusionModel(nn.Module):

    def __init__(self, architecture_class, num_classes, num_channels, hyperparameters):
        super().__init__()
        num_filters = hyperparameters["num_filters"]
        self.encoder = architecture_class(num_channels, num_filters)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        features, activations = self.encoder(x)
        a1, a2, a3 = activations
        logits = self.classifier(features)
        return {"logits": logits, "features": features, "a1": a1}


def saliency_multi_batch(model, val_loader, num_batches=5, target_mode="pred"):

    model.eval()
    device = next(model.parameters()).device

    total_sal = None
    batches_used = 0

    for b_idx, batch in enumerate(val_loader):
        if b_idx >= num_batches:
            break

        x = batch["sequence"].to(device).requires_grad_(True)

        out = model(x)
        logits = out["logits"]  # (B, num_classes)

        if isinstance(target_mode, tuple) and target_mode[0] == "class":
            k = int(target_mode[1])
            scalar = logits[:, k].sum()
        elif target_mode == "pred":
            idx = logits.argmax(dim=1)
            scalar = logits[torch.arange(logits.size(0), device=device), idx].sum()
        else:
            raise ValueError("target_mode must be 'pred' or ('class', k)")

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        scalar.backward()

        sal = x.grad.abs().mean(dim=0).mean(dim=-1)  # (C,)
        total_sal = sal if total_sal is None else (total_sal + sal)
        batches_used += 1

        x.requires_grad_(False)
        model.zero_grad(set_to_none=True)

    if batches_used == 0:
        raise RuntimeError("Validation loader had no batches!")

    avg_sal = total_sal / batches_used
    rel = avg_sal / (avg_sal.sum() + 1e-12)

    return avg_sal.cpu(), rel.cpu()


def apply_shifts(x, channels, shift):

    if shift == "zero":
        x[:, channels, :] = 0.
    elif shift == "noise":
        var = torch.empty(1, device=x.device).uniform_(1.0, 3.0).sqrt()
        x[:, channels, :] += torch.randn_like(x[:, channels, :]) * var
    elif shift == "saturate":
        sat = torch.empty(1, device=x.device).uniform_(1.0, 3.0)
        x[:, channels, :] = sat

    return x



plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
})


def plot_activation_heatmap(x, x_pert, acts_base, acts_pert, num_layers=1, save_path=None):

    num_panels = 1 + num_layers  # delta X + one per layer

    w = (470 / 72.27) * 0.95
    h = w * 0.25
    fig, axes = plt.subplots(1, num_panels, figsize=(w, h))

    def get_delta_lims(delta_tensor):
        m = torch.abs(delta_tensor).max().item()
        return -m, m

    # Delta X
    dx = x_pert - x
    v_min, v_max = get_delta_lims(dx)
    axes[0].imshow(dx[0].cpu(), aspect='auto', cmap='RdBu_r', vmin=v_min, vmax=v_max, interpolation="none")
    axes[0].set_title(r"$\Delta \mathbf{X}$")
    axes[0].set_ylabel("Channel")
    axes[0].set_xlabel("Timestep")

    # Delta A_i for each layer
    for i in range(num_layers):
        diff = acts_pert[i][0].cpu() - acts_base[i][0].cpu()
        vd_min, vd_max = get_delta_lims(diff)
        axes[i + 1].imshow(diff, aspect='auto', cmap='RdBu_r', vmin=vd_min, vmax=vd_max, interpolation="none")
        axes[i + 1].set_title(r"$\Delta \mathbf{A}_{" + str(i + 1) + r"}$")
        axes[i + 1].set_ylabel("Filter")
        axes[i + 1].set_xlabel("Timestep")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()


if __name__ == "__main__":

    dataset = "dsads"

    os.makedirs("outputs/plots", exist_ok=True)

    config = {"dataset": dataset,
              "hyperparameters": {"lr": 5e-3, "wd": 1e-5, "num_epochs": 500, "batch_size": 64, "num_filters": 8}}

    seed = 42069
    set_seed(seed)

    splits = get_loso_splits(dataset)
    split = splits["split_0"]

    target_set = split["test"]
    source_set = split["train"]

    source_train_loader, source_val_loader, target_loader = load_dataloaders(dataset, source_set, target_set,
                                                                             device="cpu")
    num_channels = source_val_loader.dataset.num_channels
    num_classes = source_train_loader.dataset.num_classes

    encoder_class = CNNEncoder
    model_class = EarlyFusionModel
    trainer_class = EarlyERMTrainer

    model = model_class(encoder_class, num_classes, num_channels, config["hyperparameters"])
    alg = trainer_class(model, num_classes, num_channels, "cpu", config["hyperparameters"])
    training_stats = alg.train(source_train_loader, source_val_loader, target_loader)

    sal_abs, sal_rel = saliency_multi_batch(
        model=alg.model,
        val_loader=source_val_loader,
        num_batches=len(source_val_loader),
        target_mode="pred",
    )

    max_k = 2
    topk = torch.topk(sal_rel, k=max_k).indices
    top_channel = topk[0]

    alg.model.eval()

    with torch.inference_mode():

        for batch in source_val_loader:
            x = batch["sequence"]

            for i in range(1, 4):

                x_ = x[i-1: i].clone()

                x_pert = x_.clone()

                # apply perturbation
                sat = torch.tensor(0.0)
                x_pert[:, top_channel, :] = sat

                # forward passes
                out_base = alg.model(x_)
                out_pert = alg.model(x_pert)

                acts_base = [out_base["a1"]]
                acts_pert = [out_pert["a1"]]

                plot_activation_heatmap(x_, x_pert, acts_base, acts_pert, save_path="outputs/plots/early.pdf")

            break

