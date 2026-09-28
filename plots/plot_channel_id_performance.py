import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import NamedTuple

import plots.plot_style as ps


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

ps.use_paper_style()

MAIN_DIR = Path("outputs/configs/main")
OUT_DIR = Path("outputs/plots")

# Local to this plot: light-neutral bar colour for per-channel results.
CH_COLOUR = "#c8c8c8"

H_MULTIPLIER = 0.25


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


def _load_across_domain(dataset_key, algo: Algorithm):
    """Return the across_domain metrics dict for the config matching
    {dataset}_{key}_*.yaml, or None if not found."""
    pattern = str(MAIN_DIR / f"{dataset_key}_{algo.key}_*.yaml")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  [warn] no config for {dataset_key}_{algo.key}")
        return None
    if len(matches) > 1:
        print(f"  [warn] multiple configs for {dataset_key}_{algo.key}, using: "
              f"{Path(matches[-1]).name}")
    with open(matches[-1]) as f:
        return yaml.safe_load(f)["metrics"]["across_domain"]


def load_aggregate(dataset_key, algo: Algorithm, metric, prefix):
    ad = _load_across_domain(dataset_key, algo)
    if ad is None:
        return None
    return _scalar(ad.get(f"{prefix}{metric}_mean"))


def count_channels(dataset_key, late_algo: Algorithm, channel_metric, prefix):
    ad = _load_across_domain(dataset_key, late_algo)
    if ad is None:
        return 0
    suffix = f"_{channel_metric}_mean_mean"
    idxs = []
    for k in ad:
        if k.startswith(prefix) and k.endswith(suffix):
            mid = k[len(prefix):-len(suffix)]
            if mid.isdigit():
                idxs.append(int(mid))
    return (max(idxs) + 1) if idxs else 0


def load_channel(dataset_key, late_algo: Algorithm, ch, channel_metric, prefix):
    ad = _load_across_domain(dataset_key, late_algo)
    if ad is None:
        return None
    return _scalar(ad.get(f"{prefix}{ch}_{channel_metric}_mean_mean"))


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def make_per_channel(late_algo, middle_algo, early_algo, metric_label, ds_map,
                     filename, prefix="test_", channel_metric="macro_f1",
                     aggregate_metric="macro_f1_mean", ncols=4):

    ds_keys = list(ds_map.keys())
    n_cols = min(ncols, len(ds_keys))
    n_rows = int(np.ceil(len(ds_keys) / n_cols))

    w = (470 / 72.27) * 0.95
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(w, w * H_MULTIPLIER * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    ch_colour = CH_COLOUR
    late_colour = ps.colour_for("late")
    middle_colour = ps.colour_for("middle")
    early_colour = ps.colour_for("early")

    for i, ds_key in enumerate(ds_keys):
        ax = axes_flat[i]

        # Normalisation reference: early's aggregate for this dataset.
        early_mean = load_aggregate(ds_key, early_algo, aggregate_metric, prefix)
        norm = early_mean if (early_mean not in (None, 0)) else np.nan

        # Per-channel bars (relative to early).
        n_ch = count_channels(ds_key, late_algo, channel_metric, prefix)
        ch_means = []
        for ch in range(n_ch):
            m = load_channel(ds_key, late_algo, ch, channel_metric, prefix)
            ch_means.append((m / norm) if m is not None else np.nan)
        ch_means = np.asarray(ch_means, dtype=float)

        x_ch = np.arange(n_ch)
        ax.bar(x_ch, ch_means, color=ch_colour, alpha=0.85, label=r"\texttt{channel}")

        # Aggregate reference bars, after a one-slot gap (relative to early).
        late_mean = load_aggregate(ds_key, late_algo,   aggregate_metric, prefix)
        middle_mean = load_aggregate(ds_key, middle_algo, aggregate_metric, prefix)
        late_mean = (late_mean / norm) if late_mean is not None else None
        middle_mean = (middle_mean / norm) if middle_mean is not None else None
        early_val = 1.0 if not np.isnan(norm) else None

        x_late = n_ch + 2.5
        x_middle = n_ch + 1.5
        x_early = n_ch + 0.5

        ax.bar(x_early, early_val if early_val is not None else np.nan,
               color=early_colour, alpha=0.85, label=r"\texttt{early}")
        ax.bar(x_middle, middle_mean if middle_mean is not None else np.nan,
               color=middle_colour, alpha=0.85, label=r"\texttt{middle}")
        ax.bar(x_late, late_mean if late_mean is not None else np.nan,
               color=late_colour, alpha=0.85, label=r"\texttt{late}")
        # Reference line at 1.0 (early).
        # ax.axhline(1.0, color=early_colour, linewidth=0.8, linestyle="--",
        #            alpha=0.6, zorder=0)

        ax.set_xticks([])
        ps.style_axes(ax)  # grid, spines, shared decimals
        ax.set_title(ds_map[ds_key])

        if i % n_cols == 0:
            ax.set_ylabel(metric_label, rotation=0, labelpad=5, ha="right", va="center")
            # ax.set_ylabel(metric_label, rotation=0, labelpad=15)
        else:
            ax.set_yticklabels([])

        ax.set_yticks(np.arange(0.0, 1.01, 0.25))  # 0, 0.25, 0.5, 0.75, 1.0

    for ax in axes_flat[len(ds_keys):]:
        ax.set_visible(False)

    # De-duplicate legend entries (each subplot re-adds the same four labels).
    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    ps.add_figure_legend(fig, uniq_h, uniq_l, y=0.025)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"{filename}.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print(f"saved at {file_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    EARLY = Algorithm("erm_early_cnn",  "early",  r"\texttt{early}")
    MIDDLE = Algorithm("erm_middle_cnn", "middle", r"\texttt{middle}")
    LATE = Algorithm("erm_late_cnn",   "late",   r"\texttt{late}")

    DS_MAP = {
        "dsads":   r"\texttt{DSADS}",
        "mhealth": r"\texttt{MHEALTH}",
        "pamap":   r"\texttt{PAMAP}",
        "wisdm":   r"\texttt{WISDM}",
        "bnci1":   r"\texttt{BNCI-1}",
        "bnci2":   r"\texttt{BNCI-2}",
        "bnci4":   r"\texttt{BNCI-4}",
        "zhou":    r"\texttt{ZHOU}",
    }

    make_per_channel(
        late_algo=LATE,
        middle_algo=MIDDLE,
        early_algo=EARLY,
        metric_label=r"$\frac{m^{\texttt{ID}}}{m^{\texttt{ID}}_{\texttt{early}}}$",
        ds_map=DS_MAP,
        filename="per_channel_id_relative",
        prefix="val_",
        ncols=4,
    )