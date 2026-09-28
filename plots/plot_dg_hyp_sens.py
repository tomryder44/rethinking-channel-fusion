import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import plots.plot_style as ps


ps.use_paper_style()

DG_DIR = Path("outputs/configs/dg")
OUT_DIR = Path("outputs/plots")

H_MULTIPLIER = 0.25

# Which DG algorithm to inspect.
ALGO = "diversify"

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

METRICS = [
    {"test_macro_f1_mean": r"$m_{\text{avg}}^{\texttt{OOD}}$"},
    {"test_macro_f1_min":  r"$m_{\text{worst}}^{\texttt{OOD}}$"},
]

# Up to 5 configs; one colour + marker each (config index -> style).
CONFIG_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7570b3"]
CONFIG_MARKERS = ["o", "s", "^", "D", "v"]


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def _scalar(v):
    return v[0] if isinstance(v, list) else v


def config_paths(algo, dataset_key):
    """All config files for one algorithm/dataset, in filename order."""
    pattern = str(DG_DIR / f"{dataset_key}_{algo}_early_cnn_*.yaml")
    return sorted(glob.glob(pattern))


def n_configs(algo, ds_keys):
    """Max number of configs across the given datasets (defines the series count)."""
    return max((len(config_paths(algo, k)) for k in ds_keys), default=0)


def read_config(path, metric):
    with open(path) as f:
        across = yaml.safe_load(f)["metrics"]["across_domain"]
    return _scalar(across[f"{metric}_mean"]), _scalar(across[f"{metric}_std"])


def config_value(algo, dataset_key, cfg_idx, metric):
    """(mean, std) of `metric` for the cfg_idx-th config on this dataset, or
    (nan, 0.0) if that dataset has fewer configs."""
    paths = config_paths(algo, dataset_key)
    if cfg_idx >= len(paths):
        return np.nan, 0.0
    return read_config(paths[cfg_idx], metric)


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def _plot_config_on_ax(ax, x, algo, cfg_idx, ds_keys, metric_key, offset=0.0):
    means, stds = [], []
    for ds_key in ds_keys:
        mean, std = config_value(algo, ds_key, cfg_idx, metric_key)
        means.append(mean)
        stds.append(std)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    colour = CONFIG_COLOURS[cfg_idx % len(CONFIG_COLOURS)]
    marker = CONFIG_MARKERS[cfg_idx % len(CONFIG_MARKERS)]
    xo = np.asarray(x, dtype=float) + offset
    ax.plot(xo, means, color=colour, marker=marker, linestyle="none",
            label=rf"cfg {cfg_idx + 1}", zorder=2)
    if np.any(stds > 0):
        ax.errorbar(xo, means, yerr=stds, fmt="none", color=colour,
                    zorder=1, **ps.ERRORBAR_KW)


def make_plot(algo, metrics, filename, groups, span=0.4):
    group_items = list(groups.items())
    ds_keys_by_col = [keys for _, keys in group_items]
    group_labels = [label for label, _ in group_items]

    n_rows = len(metrics)
    n_cols = len(group_items)
    w = (470 / 72.27) * 0.95

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(w, w * H_MULTIPLIER * (0.8 * n_rows if n_rows > 1 else 1.0)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1] * n_cols},
    )

    # series count is per-column (HAR and MI may sweep different counts)
    offsets_by_col = []
    for ds_keys in ds_keys_by_col:
        n = n_configs(algo, ds_keys)
        offsets_by_col.append(np.linspace(-span / 2, span / 2, n) if n else np.array([]))

    for row, metric in enumerate(metrics):
        metric_key, metric_label = next(iter(metric.items()))
        is_last_row = (row == n_rows - 1)

        for col, (group_label, ds_keys) in enumerate(zip(group_labels, ds_keys_by_col)):
            ax = axes[row][col]
            x = np.arange(len(ds_keys))
            ds_labels = [ALL_DATASETS[k] for k in ds_keys]

            offsets = offsets_by_col[col]
            for cfg_idx, off in enumerate(offsets):
                _plot_config_on_ax(ax, x, algo, cfg_idx, ds_keys, metric_key, off)

            ax.set_xticks(x)
            ax.set_xticklabels(ds_labels if is_last_row else [""] * len(ds_labels))
            ps.style_axes(ax)

            if row == 0 and group_label is not None and n_cols > 1:
                ax.set_title(group_label)

        axes[row][0].set_ylabel(metric_label, rotation=0, labelpad=20)

    handles, labels = axes[0][0].get_legend_handles_labels()
    ps.add_figure_legend(fig, handles, labels)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"{filename}.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print(f"saved at {file_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    make_plot(
        algo=ALGO,
        metrics=METRICS,
        filename=f"dg_single_{ALGO}",
        groups=GROUPS,
    )