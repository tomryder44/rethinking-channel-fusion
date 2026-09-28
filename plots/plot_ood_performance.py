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

DIRS = {
    "main": Path("outputs/configs/main"),
    "dg":   Path("outputs/configs/dg"),
}
OUT_DIR = Path("outputs/plots")

H_MULTIPLIER = 0.25

# metric on which the best-dg config is chosen, within and across algorithms
SELECT_METRIC = "test_macro_f1_mean"

# DG algorithms to consider for the best-dg reference (folder: dg/)
DG_ALGORITHMS = ["diversify", "groupdro", "irm", "mmd", "nnr", "phaser", "vrex"]

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
    is_best_dg: bool = False  # if True, value comes from the DG selection, not a single config


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def _scalar(v):
    """Unwrap single-element lists that yaml sometimes produces."""
    if isinstance(v, list):
        return v[0]
    return v


def _read_metric(data, metric):
    """Pull (mean, std) for a metric out of an already-loaded config."""
    across = data["metrics"]["across_domain"]
    return _scalar(across[f"{metric}_mean"]), _scalar(across[f"{metric}_std"])


def load_metric(dataset_key, algo: Algorithm, metric):
    """Load (mean, std) for a metric from the config matching {dataset}_{key}_*.yaml.

    Returns (None, None) if no config is found.
    """
    pattern = str(DIRS["main"] / f"{dataset_key}_{algo.key}_*.yaml")
    matches = sorted(glob.glob(pattern))

    if not matches:
        print(f"  [warn] no config for {dataset_key}_{algo.key}")
        return None, None
    if len(matches) > 1:
        print(f"  [warn] multiple configs for {dataset_key}_{algo.key}, using: "
              f"{Path(matches[-1]).name}")

    with open(matches[-1]) as f:
        data = yaml.safe_load(f)
    return _read_metric(data, metric)


def _best_config_for_algorithm(dataset_key, algo_name):
    """Among a DG algorithm's configs (one per swept hyperparameter value), return
    (best_mean, best_path, best_data) ranked by SELECT_METRIC's mean. (None, ...) if none."""
    pattern = str(DIRS["dg"] / f"{dataset_key}_{algo_name}_early_cnn_*.yaml")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  [warn] no dg configs for {dataset_key}_{algo_name}")
        return None, None, None

    best = (None, None, None)  # (mean, path, data)
    for m in matches:
        with open(m) as f:
            data = yaml.safe_load(f)
        mean, _ = _read_metric(data, SELECT_METRIC)
        if best[0] is None or mean > best[0]:
            best = (mean, m, data)
    return best


def select_best_dg(dataset_key, metric):
    """Two-level selection for the best-dg reference on one dataset.

    Within each algorithm: pick the config with the highest SELECT_METRIC mean.
    Across algorithms: pick the algorithm whose winning config is highest.
    Prints both levels. Returns (mean, std) of `metric` for the champion config.
    """
    per_algo_winners = []  # (mean, algo_name, path, data)
    for algo_name in DG_ALGORITHMS:
        best_mean, best_path, best_data = _best_config_for_algorithm(dataset_key, algo_name)
        if best_mean is None:
            continue
        print(f"  [{dataset_key}] {algo_name}: winner = {Path(best_path).name} "
              f"({SELECT_METRIC}_mean={best_mean:.4f})")
        per_algo_winners.append((best_mean, algo_name, best_path, best_data))

    if not per_algo_winners:
        print(f"  [{dataset_key}] best-dg: no DG configs found")
        return None, None

    champ_mean, champ_algo, champ_path, champ_data = max(per_algo_winners, key=lambda t: t[0])
    print(f"  [{dataset_key}] best-dg = {champ_algo} @ {Path(champ_path).name} "
          f"({SELECT_METRIC}_mean={champ_mean:.4f})")

    # selection is always on SELECT_METRIC, but we plot `metric` from the champion
    return _read_metric(champ_data, metric)


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def _series_for_algo(algo, ds_keys, metric_key):
    """Return (means, stds) arrays for one algorithm across ds_keys."""
    means, stds = [], []
    for ds_key in ds_keys:
        if algo.is_best_dg:
            mean, std = select_best_dg(ds_key, metric_key)
        else:
            mean, std = load_metric(ds_key, algo, metric_key)
        means.append(mean if mean is not None else np.nan)
        stds.append(std if std is not None else 0.0)
    return means, stds


def _plot_algo_on_ax(ax, x, algo, ds_keys, metric_key, offset=0.0):
    means, stds = _series_for_algo(algo, ds_keys, metric_key)
    ps.plot_series(ax, x, means, algo.method, algo.label, stds=stds, offset=offset)


def make_plot(algorithms, metrics, filename, groups, span=0.35):
    if isinstance(metrics, dict):
        metrics = [metrics]

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

    offsets = ps.offsets_for(len(algorithms), span=span)

    for row, metric in enumerate(metrics):
        metric_key, metric_label = next(iter(metric.items()))
        is_last_row = (row == n_rows - 1)

        for col, (group_label, ds_keys) in enumerate(zip(group_labels, ds_keys_by_col)):
            ax = axes[row][col]
            x = np.arange(len(ds_keys))
            ds_labels = [ALL_DATASETS[k] for k in ds_keys]

            for algo, off in zip(algorithms, offsets):
                _plot_algo_on_ax(ax, x, algo, ds_keys, metric_key, off)

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
        algorithms=[
            Algorithm("erm_early_cnn",  "early",  r"\texttt{early}"),
            Algorithm("erm_middle_cnn", "middle", r"\texttt{middle}"),
            Algorithm("erm_late_cnn",   "late",   r"\texttt{late}"),
            Algorithm("",               "best-dg", r"\texttt{best-dg}", is_best_dg=True),
        ],
        metrics=[
            {"test_macro_f1_mean": r"$m_{\text{avg}}^{\texttt{OOD}}$"},
            {"test_macro_f1_min":  r"$m_{\text{worst}}^{\texttt{OOD}}$"},
        ],
        filename="ood_performance",
        groups=GROUPS,
    )