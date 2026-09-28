import glob
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import NamedTuple

import plots.plot_style as ps


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

ps.use_paper_style()

MAIN_DIR = Path("outputs/configs/main")
OUT_DIR = Path("outputs/plots")

H_MULTIPLIER = 0.25

ALL_DATASETS = {
    "dsads":   r"\texttt{DSADS}",
    "mhealth": r"\texttt{MHEALTH}",
    "pamap":   r"\texttt{PAMAP}",
    "wisdm":   r"\texttt{WISDM}",
    "bnci1":   r"\texttt{BNCI-1}",
    "bnci2":   r"\texttt{BNCI-2}",
    "bnci4":   r"\texttt{BNCI-4}",
    "zhou":    r"\texttt{ZHOU}",
}

GROUPS = {
    "HAR": ["dsads", "mhealth", "pamap", "wisdm"],
    "MI":  ["bnci1", "bnci2", "bnci4", "zhou"],
}


class Algorithm(NamedTuple):
    key: str     # {algorithm}_{fusion}_{encoder}, matched against the filename
    method: str  # canonical method name, used for the fixed colour lookup
    label: str   # LaTeX legend label


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def _scalar(v):
    """Unwrap single-element lists that yaml sometimes produces."""
    if isinstance(v, list):
        return v[0]
    return v


def load_metric(dataset_key, algo: Algorithm, metric):
    """Load (mean, std) for a metric from the config matching {dataset}_{key}_*.yaml.

    Returns (None, None) if no config is found.
    """
    pattern = str(MAIN_DIR / f"{dataset_key}_{algo.key}_*.yaml")
    matches = sorted(glob.glob(pattern))

    if not matches:
        print(f"  [warn] no config for {dataset_key}_{algo.key}")
        return None, None
    if len(matches) > 1:
        print(f"  [warn] multiple configs for {dataset_key}_{algo.key}, using: "
              f"{Path(matches[-1]).name}")

    with open(matches[-1]) as f:
        data = yaml.safe_load(f)

    across = data["metrics"]["across_domain"]
    mean = _scalar(across[f"{metric}_mean"])
    std = _scalar(across[f"{metric}_std"])
    return mean, std


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def _plot_algo_on_ax(ax, x, algo, ds_keys, metric_key, offset=0.0):
    means, stds = [], []
    for ds_key in ds_keys:
        mean, std = load_metric(ds_key, algo, metric_key)
        means.append(mean if mean is not None else np.nan)
        stds.append(std if std is not None else 0.0)
    ps.plot_series(ax, x, means, algo.method, algo.label, stds=stds, offset=offset)


def make_plot(algorithms, metric, filename, groups):
    metric_key, metric_label = next(iter(metric.items()))

    group_items = list(groups.items())
    ds_keys_by_col = [keys for _, keys in group_items]
    group_labels = [label for label, _ in group_items]

    n_cols = len(group_items)
    w = (470 / 72.27) * 0.95

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(w, w * H_MULTIPLIER),
        squeeze=False,
        gridspec_kw={"width_ratios": [1] * n_cols},
    )

    for col, (group_label, ds_keys) in enumerate(zip(group_labels, ds_keys_by_col)):
        ax = axes[0][col]
        x = np.arange(len(ds_keys))
        ds_labels = [ALL_DATASETS[k] for k in ds_keys]

        offsets = ps.offsets_for(len(algorithms), span=0.3)
        for algo, off in zip(algorithms, offsets):
            _plot_algo_on_ax(ax, x, algo, ds_keys, metric_key, off)

        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels)
        ps.style_axes(ax)

        if group_label is not None and n_cols > 1:
            ax.set_title(group_label)

    # axes[0][0].set_ylabel(metric_label, rotation=0, labelpad=20)
    axes[0][0].set_ylabel(metric_label, rotation=0, labelpad=5, ha="right", va="center")

    handles, labels = axes[0][0].get_legend_handles_labels()
    ps.add_figure_legend(fig, handles, labels)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"{filename}.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print(f"saved at {file_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    make_plot(
        algorithms=[
            Algorithm("erm_early_cnn",  "early",  r"\texttt{early}"),
            Algorithm("erm_middle_cnn", "middle", r"\texttt{middle}"),
            Algorithm("erm_late_cnn",   "late",   r"\texttt{late}"),
        ],
        metric={"val_macro_f1_mean": r"$m_{\text{avg}}^{\texttt{ID}}$"},
        filename="id_performance",
        groups=GROUPS,
    )