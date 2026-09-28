import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import plots.plot_style as ps


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

ps.use_paper_style()

CONFIG_DIR = Path("outputs/configs/sensor_corruptions")
OUT_DIR = Path("outputs/plots")

DATASETS = ["dsads", "mhealth", "pamap"]

# Row = corruption granularity; column = shift type.
GRANS = ["channel", "sensor"]
GRAN_LABELS = {
    "channel": r"channel",
    "sensor":  r"sensor",
}
SHIFTS = ["zero", "saturate", "noise"]

# Fusion series (config keys) and their canonical method name for colour lookup.
FUSIONS = [
    ("early",         "early"),
    ("early_erm_ens", "early-ens-C"),
    ("middle",        "middle"),
    ("middle_en",     "middle-en"),
    ("late",          "late"),
]
LABELS = {
    "early":         r"\texttt{early}",
    "early_erm_ens": r"\texttt{early-ens}",
    "middle":        r"\texttt{middle}",
    "middle_en":     r"\texttt{middle-en}",
    "late":          r"\texttt{late}",
}


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def load_results(dataset_key):
    """Return results[gran][fusion][shift] structure for {dataset}_*.yaml."""
    pattern = str(CONFIG_DIR / f"{dataset_key}_*.yaml")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  [warn] no config for {dataset_key}")
        return None
    if len(matches) > 1:
        print(f"  [warn] multiple configs for {dataset_key}, using: "
              f"{Path(matches[-1]).name}")
    with open(matches[-1]) as f:
        return yaml.safe_load(f)["results"]


def _series(node):
    """node maps k -> list of per-run floats (k may be int or str).

    Returns (ks, means, stds); std is across runs (0 for a single run).
    """
    ks = sorted(node.keys(), key=int)
    means, stds = [], []
    for k in ks:
        vals = np.asarray(node[k], dtype=float)
        means.append(vals.mean())
        stds.append(vals.std())
    return np.array([int(k) for k in ks]), np.asarray(means), np.asarray(stds)


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def make_corruption_plot(dataset):

    results = load_results(dataset)

    grans = [g for g in GRANS if g in results]

    w = (470 / 72.27) * 0.95
    fig, axes = plt.subplots(
        nrows=len(grans), ncols=len(SHIFTS),
        figsize=(w, 2.25 * len(grans) / len(GRANS)),
        sharey="row", sharex=False,
        squeeze=False,
    )

    for r, gran in enumerate(grans):
        for c, sh in enumerate(SHIFTS):
            ax = axes[r][c]
            ks = None
            for fusion_key, method in FUSIONS:
                node = results[gran][fusion_key][sh]
                ks, ys, ystd = _series(node)
                ps.plot_trend(ax, ks, ys, method, LABELS[fusion_key], stds=ystd)

            ps.style_axes(ax)
            ax.set_xticks(ks)
            if gran == "channel":
                ax.set_xticklabels([str(k) if k in (ks[0], ks[-1]) else "" for k in ks])

            # X label on the bottom row only.
            if r == len(grans) - 1:
                ax.set_xlabel("No. corrupted")

            # Y label once, on the left edge.
            if c == 0 and r == 0:
                ax.set_ylabel("Macro F1", rotation=90, labelpad=5, ha="right", va="center")
                ax.yaxis.set_label_coords(-0.28, 0.0)

            # Shift name as column title (top row).
            if r == 0:
                ax.set_title(sh, pad=6)

            # Granularity as a right-side label on the last column.
            if c == len(SHIFTS) - 1:
                ax_right = ax.twinx()
                ax_right.set_ylabel(GRAN_LABELS[gran], rotation=270, labelpad=12)
                ax_right.set_yticks([])
                ax_right.spines[["top", "right", "left", "bottom"]].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    ps.add_figure_legend(fig, handles, labels, y=-0.075)

    plt.subplots_adjust(wspace=0.1, hspace=0.35)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"sensor_corruptions_{dataset}.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print(f"saved at {file_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    for dataset in DATASETS:
        make_corruption_plot(dataset)