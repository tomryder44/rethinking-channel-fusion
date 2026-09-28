import glob
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

import plots.plot_style as ps


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

ps.use_paper_style()

CONFIG_DIR = Path("outputs/configs/late_weights")
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

# Per-dataset line colours (these are datasets, not methods, so use the cycle).
CYCLE = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]

_TAU_RE = re.compile(r"^tau_([\d.]+)_mean_mean$")


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def load_across_domain(dataset_key):
    """Return the results.across_domain dict for {dataset}_*.yaml, or None."""
    pattern = str(CONFIG_DIR / f"{dataset_key}_*.yaml")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  [warn] no config for {dataset_key}")
        return None
    if len(matches) > 1:
        print(f"  [warn] multiple configs for {dataset_key}, using: "
              f"{Path(matches[-1]).name}")
    with open(matches[-1]) as f:
        return yaml.safe_load(f)["results"]["across_domain"]


def load_tau_sweep(dataset_key, reduction="mean"):
    """Return (taus, means, stds) sorted by tau, for reduction in {mean, worst}."""
    ad = load_across_domain(dataset_key)
    if ad is None:
        return None, None, None

    triples = []
    for k in ad:
        m = _TAU_RE.match(k)
        if not m:
            continue
        tau = float(m.group(1))
        val = ad[f"tau_{m.group(1)}_{reduction}_mean"]
        std = ad.get(f"tau_{m.group(1)}_{reduction}_std", 0.0)
        triples.append((tau, val, std))

    triples.sort(key=lambda t: t[0])
    taus = np.array([t[0] for t in triples])
    means = np.array([t[1] for t in triples])
    stds = np.array([t[2] for t in triples])
    return taus, means, stds


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def _plot_dataset(ax, taus, means, stds):
    ax.plot(taus, means, linewidth=1.5, linestyle="-", zorder=3)
    if np.any(stds > 0):
        ax.fill_between(taus, means - stds, means + stds, alpha=0.15, zorder=2)


def make_plot(groups, metric_label, filename, reduction="mean"):
    group_items = list(groups.items())
    ds_keys_by_row = [keys for _, keys in group_items]
    n_rows = len(group_items)
    n_cols = max(len(keys) for keys in ds_keys_by_row)

    w = (470 / 72.27) * 0.95
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(w, w * H_MULTIPLIER * n_rows),
        squeeze=False,
    )

    for row, ds_keys in enumerate(ds_keys_by_row):
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(ds_keys):
                ax.set_visible(False)
                continue

            ds_key = ds_keys[col]
            taus, means, stds = load_tau_sweep(ds_key, reduction=reduction)
            if taus is not None:
                _plot_dataset(ax, taus, means, stds)

            tau_star = load_tau_star(ds_key)
            if tau_star is not None:
                ax.axvline(tau_star, color="0.4", linewidth=0.9,
                           linestyle=":", alpha=0.8, zorder=1)

            ax.set_xscale("log")
            ax.set_title(ALL_DATASETS[ds_key])
            if row == len(ds_keys_by_row) - 1:
                ax.set_xlabel(r"$\tau$")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ps.style_axes(ax)  # grid, spines, shared decimals
            # log-scale minor grid + integer-ish tau tick labels
            ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:g}$"))

            if col == 0:
                ax.set_ylabel(metric_label, rotation=0, labelpad=15)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35, hspace=0.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"{filename}.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print(f"saved at {file_path}")


def load_tau_star(dataset_key):
    ad = load_across_domain(dataset_key)
    if ad is None:
        return None
    return ad.get("tau_star_mean_mean")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    make_plot(GROUPS, r"$m_{\text{avg}}^{\text{OOD}}$",
              "tau_sweep_avg", reduction="mean")

    # make_plot(GROUPS, r"$m_{\text{worst}}^{\text{OOD}}$",
    #           "tau_sweep_worst", reduction="worst")