"""Fusion-selection OOD performance + selection proportions (nested LOSO-CV).

Three rows:
  1. average OOD performance (five methods: early, middle, late, oracle, nested)
  2. worst-group OOD performance (same five)
  3. selection proportion per dataset (early/middle/late; stacked, sums to 1)

Rows 1-2 use the dodge/markers style (five overlapping methods). Row 3 is a
stacked bar since the proportions are a composition. Configs are one per dataset
as {dataset}_*.yaml under fusion_selection; metrics live flat under `results`.
"""

import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import plots.plot_style as ps


ps.use_paper_style()

CONFIG_DIR = Path("outputs/configs/fusion_selection")
OUT_DIR = Path("outputs/plots")

H_MULTIPLIER = 0.25

ALL_DATASETS = {
    "dsads": r"\texttt{DSADS}",
    "mhealth": r"\texttt{MHEALTH}",
    "pamap": r"\texttt{PAMAP}",
    "wisdm": r"\texttt{WISDM}",
    "bnci1": r"\texttt{BNCI-1}",
    "bnci2": r"\texttt{BNCI-2}",
    "bnci4": r"\texttt{BNCI-4}",
    "zhou": r"\texttt{ZHOU}",
}

GROUPS = {
    "HAR": ["dsads", "mhealth", "pamap", "wisdm"],
    "MI": ["bnci1", "bnci2", "bnci4", "zhou"],
}

# Methods on the performance rows, in draw order.
METHODS = ["early", "middle", "late", "oracle", "nested"]
LABELS = {
    "early": r"\texttt{early}",
    "middle": r"\texttt{middle}",
    "late": r"\texttt{late}",
    "oracle": r"oracle",
    "nested": r"selection",
}
# Fusions that make up the selection proportion (row 3).
PROP_FUSIONS = ["early", "middle", "late"]


def _scalar(v):
    return v[0] if isinstance(v, list) else v


def load_results(dataset_key):
    pattern = str(CONFIG_DIR / f"{dataset_key}_*.yaml")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  [warn] no config for {dataset_key}")
        return None
    if len(matches) > 1:
        print(f"  [warn] multiple configs for {dataset_key}, using: {Path(matches[-1]).name}")
    with open(matches[-1]) as f:
        return yaml.safe_load(f)["results"]


def load_scores(dataset_key, reduction):
    """{method: (mean, std)} for reduction in {mean, worst}, or None."""
    r = load_results(dataset_key)
    if r is None:
        return None
    return {m: (_scalar(r[f"{m}_{reduction}_mean"]), _scalar(r[f"{m}_{reduction}_std"]))
            for m in METHODS}


def load_props(dataset_key):
    """{fusion: proportion} for early/middle/late, or None."""
    r = load_results(dataset_key)
    if r is None:
        return None
    return {f: _scalar(r[f"prop_{f}_mean"]) for f in PROP_FUSIONS}


def _plot_perf_row(ax, ds_keys, reduction):
    x = np.arange(len(ds_keys))
    scores = {k: load_scores(k, reduction) for k in ds_keys}

    n = len(METHODS)
    span = 0.4
    offsets = np.linspace(-span / 2, span / 2, n)

    for method, off in zip(METHODS, offsets):
        means = np.array([scores[k][method][0] if scores[k] else np.nan for k in ds_keys])
        stds = np.array([scores[k][method][1] if scores[k] else 0.0 for k in ds_keys])
        colour = ps.colour_for(method)
        xo = x + off
        ax.plot(xo, means, color=colour, marker=ps.marker_for(method),
                linestyle="none", label=LABELS[method], zorder=2)
        if np.any(stds > 0):
            ax.errorbar(xo, means, yerr=stds, fmt="none", color=colour,
                        zorder=1, **ps.ERRORBAR_KW)

    ax.set_xticks(x)
    return x


def _plot_prop_row(ax, ds_keys):
    x = np.arange(len(ds_keys))
    props = {k: load_props(k) for k in ds_keys}

    bottom = np.zeros(len(ds_keys))
    for fusion in PROP_FUSIONS:
        vals = np.array([props[k][fusion] if props[k] else np.nan for k in ds_keys])
        drawn = np.where(vals == 0, np.nan, vals)          # no patch for zero shares
        ax.bar(x, drawn, bottom=bottom, color=ps.colour_for(fusion),
               width=0.6, label=LABELS[fusion], zorder=2,
               linewidth=0, edgecolor="none")
        bottom += np.nan_to_num(vals)                       # still advance the stack

    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    return x


def make_plot(groups, filename):
    group_items = list(groups.items())
    ds_keys_by_col = [keys for _, keys in group_items]
    group_labels = [label for label, _ in group_items]
    n_cols = len(group_items)

    row_specs = [
        ("perf", "mean", r"$m_{\text{avg}}^{\text{OOD}}$"),
        ("perf", "worst", r"$m_{\text{worst}}^{\text{OOD}}$"),
        ("prop", None, r"prop."),
    ]
    n_rows = len(row_specs)

    w = (470 / 72.27) * 0.95
    # fig, axes = plt.subplots(n_rows, n_cols,
    #                          figsize=(w, w * H_MULTIPLIER * n_rows * 0.9),
    #                          squeeze=False,
    #                          gridspec_kw={"height_ratios": [1, 1, 0.6]})

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(w, w * H_MULTIPLIER * 2.6 * 0.8),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.6]})

    for row, (kind, reduction, ylabel) in enumerate(row_specs):
        is_last_row = (row == n_rows - 1)
        for col, ds_keys in enumerate(ds_keys_by_col):
            ax = axes[row][col]
            ds_labels = [ALL_DATASETS[k] for k in ds_keys]

            if kind == "perf":
                _plot_perf_row(ax, ds_keys, reduction)
                ps.style_axes(ax)
            else:
                _plot_prop_row(ax, ds_keys)
                ps.style_axes(ax)

            ax.set_xticklabels(ds_labels if is_last_row else [""] * len(ds_labels))
            if row == 0 and n_cols > 1:
                ax.set_title(group_labels[col])

        axes[row][0].set_ylabel(ylabel, rotation=0, labelpad=5, ha="right", va="center")

    handles, labels = axes[0][0].get_legend_handles_labels()
    ps.add_figure_legend(fig, handles, labels)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"{filename}.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print(f"saved at {file_path}")


if __name__ == "__main__":
    make_plot(GROUPS, "fusion_selection")