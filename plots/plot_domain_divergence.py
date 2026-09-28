import glob
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# Sigma sweep per group (different scales for HAR vs MI).
SIGMAS = {
    "HAR": [2, 4, 8, 16],
    "MI":  [0.5, 1, 2, 4],
}

# Which group each dataset belongs to (for sigma lookup).
GROUP_OF = {ds: g for g, keys in GROUPS.items() for ds in keys}


class Algorithm(NamedTuple):
    key: str     # {algorithm}_{fusion}_{encoder}, matched against the filename
    method: str  # canonical method name, used for the fixed colour lookup
    label: str   # LaTeX legend label


ALGOS = [
    Algorithm("erm_early_cnn",  "early",  r"\texttt{early}"),
    Algorithm("erm_middle_cnn", "middle", r"\texttt{middle}"),
]


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def _scalar(v):
    """Unwrap single-element lists that yaml sometimes produces."""
    if isinstance(v, list):
        return v[0]
    return v


def read(ds_key, algo: Algorithm, metric, agg="mean"):
    """Load ({metric}_{agg}_mean, _std) from the config matching {ds}_{key}_*.yaml.

    Returns (nan, 0.0) if no config is found, so a missing series is visibly
    absent rather than crashing.
    """
    pattern = str(MAIN_DIR / f"{ds_key}_{algo.key}_*.yaml")
    matches = sorted(glob.glob(pattern))

    if not matches:
        print(f"  [warn] no config for {ds_key}_{algo.key}")
        return np.nan, 0.0
    if len(matches) > 1:
        print(f"  [warn] multiple configs for {ds_key}_{algo.key}, using: "
              f"{Path(matches[-1]).name}")

    with open(matches[-1]) as f:
        across = yaml.safe_load(f)["metrics"]["across_domain"]

    mean = _scalar(across.get(f"{metric}_{agg}_mean", np.nan))
    std = _scalar(across.get(f"{metric}_{agg}_std", 0.0))
    return mean, std


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def _plot_trend(ax, x, means, stds, algo: Algorithm):
    ps.plot_trend(ax, x, means, algo.method, algo.label, stds=stds)


def _annotate_median(ax, ds_key):
    short = {"early": r"\texttt{e}", "middle": r"\texttt{m}"}
    for i, algo in enumerate(ALGOS):
        mean, _ = read(ds_key, algo, "test_median_pairwise", agg="mean")
        ax.text(
            0.95, 1 - i * 0.13, rf"{short[algo.method]}: {mean:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            color=ps.colour_for(algo.method),
            fontsize=plt.rcParams["font.size"] * 0.75, linespacing=1.3,
        )


# --------------------------------------------------------------------------- #
# Main figure: Gaussian-MMD vs sigma (2 rows x 4 cols, HAR then MI)           #
# --------------------------------------------------------------------------- #

def main():
    group_items = list(GROUPS.items())          # [("HAR", [...]), ("MI", [...])]
    n_rows = len(group_items)                    # 2
    n_cols = max(len(keys) for _, keys in group_items)  # 4

    w = (470 / 72.27) * 0.95
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(w, w * H_MULTIPLIER * n_rows),
        squeeze=False,
    )

    for row, (group_label, ds_keys) in enumerate(group_items):
        sigmas = SIGMAS[group_label]
        for col in range(n_cols):
            ax = axes[row][col]

            if col >= len(ds_keys):
                ax.set_visible(False)
                continue

            ds_key = ds_keys[col]

            for algo in ALGOS:
                means = np.array([read(ds_key, algo, f"test_gmmd_{s}")[0] for s in sigmas])
                errs = np.array([read(ds_key, algo, f"test_gmmd_{s}")[1] for s in sigmas])
                _plot_trend(ax, sigmas, means, errs, algo)

            _annotate_median(ax, ds_key)

            ax.set_xscale("log", base=2)
            ax.set_xticks(sigmas)
            ax.set_xticklabels([f"{s:g}" for s in sigmas])
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))
            ps.style_axes(ax)  # grid, spines, shared decimals
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax.set_xlabel(r"$\sigma$")
            ax.set_title(ALL_DATASETS[ds_key])

        axes[row][0].set_ylabel(r"$\text{Div}_{\text{avg}}^{\mathcal{M}_\mathrm{g}}$",
                                rotation=0, labelpad=5, ha="right", va="center")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.6, hspace=0.75)
    ps.add_figure_legend(fig, handles, labels, y=0.025)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / "domain_divergence.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print(f"saved at {file_path}")


APP_METRICS = [
    ("test_wasserstein", r"$\text{Div}_{\text{avg}}^{\mathcal{W}}$"),
    ("test_emmd",        r"$\text{Div}_{\text{avg}}^{\mathcal{M}_\mathrm{e}}$"),
]


def appendix():
    group_items = list(GROUPS.items())
    group_labels = [label for label, _ in group_items]
    ds_keys_by_col = [keys for _, keys in group_items]

    n_rows = len(APP_METRICS)
    n_cols = len(group_items)
    w = (470 / 72.27) * 0.95

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(w, w * H_MULTIPLIER * (0.8 * n_rows if n_rows > 1 else 1.0)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1] * n_cols},
    )

    offsets = ps.offsets_for(len(ALGOS), span=0.15)

    for row, (metric, metric_label) in enumerate(APP_METRICS):
        is_last_row = (row == n_rows - 1)

        for col, (group_label, ds_keys) in enumerate(zip(group_labels, ds_keys_by_col)):
            ax = axes[row][col]
            x = np.arange(len(ds_keys))
            ds_labels = [ALL_DATASETS[k] for k in ds_keys]

            for algo, off in zip(ALGOS, offsets):
                means = np.array([read(ds_key, algo, metric)[0] for ds_key in ds_keys])
                errs = np.array([read(ds_key, algo, metric)[1] for ds_key in ds_keys])
                ps.plot_series(ax, x, means, algo.method, algo.label,
                               stds=errs, offset=off)

            ax.set_xticks(x)
            ax.set_xticklabels(ds_labels if is_last_row else [""] * len(ds_labels))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))
            ps.style_axes(ax)  # grid, spines, shared decimals

            if row == 0 and n_cols > 1:
                ax.set_title(group_label)

        axes[row][0].set_ylabel(metric_label, rotation=0, labelpad=5,
                                ha="right", va="center")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    ps.add_figure_legend(fig, handles, labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / "domain_divergence_appendix.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print(f"saved at {file_path}")


if __name__ == "__main__":
    main()
    appendix()