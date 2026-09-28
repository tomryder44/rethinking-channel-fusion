import glob
import re
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

DIRS = {
    "main": Path("outputs/configs/main"),
    "early_ens": Path("outputs/configs/early_ens"),
    "dg": Path("outputs/configs/dg"),
    "middle_variants": Path("outputs/configs/middle_variants"),
}

OUT_DIR = Path("outputs/plots")

DATASETS = ["dsads", "mhealth", "pamap", "wisdm", "bnci1", "bnci2", "bnci4", "zhou"]

# LaTeX row labels for the datasets (shown down the left edge).
DATASET_LABELS = {
    "dsads":   r"\texttt{DSADS}",
    "mhealth": r"\texttt{MHEALTH}",
    "pamap":   r"\texttt{PAMAP}",
    "wisdm":   r"\texttt{WISDM}",
    "bnci1":   r"\texttt{BNCI-1}",
    "bnci2":   r"\texttt{BNCI-2}",
    "bnci4":   r"\texttt{BNCI-4}",
    "zhou":    r"\texttt{ZHOU}",
}

# base metric to build the per-subject distribution from (NO _mean/_std suffix):
# per_domain stores `{i}_{SUBJECT_METRIC}` (per-seed list),
# `{i}_{SUBJECT_METRIC}_mean` (seed-averaged) and `{i}_{SUBJECT_METRIC}_std`
# (across-seed std) for each held-out subject i.
SUBJECT_METRIC = "test_macro_f1"
SUBJECT_METRIC_LABEL = r"$m^{\texttt{OOD}}_d$"

# best-dg reference: champion is chosen on this across-domain metric, two levels
# (best config within each DG algorithm, then best algorithm across them). The
# per-subject distribution is then read from that champion config.
SELECT_METRIC = "test_macro_f1_mean"
DG_ALGORITHMS = ["diversify", "groupdro", "irm", "mmd", "nnr", "phaser", "vrex"]

SPREAD = 0.35


class Algorithm(NamedTuple):
    key: str     # {algorithm}_{fusion}_{encoder}, matched against the filename
    method: str  # canonical method name, used for the fixed colour lookup
    label: str   # LaTeX legend label
    dir: str = "main"            # config dir key
    n_models: str | None = None  # ensemble size selector; None for non-ensemble
    is_best_dg: bool = False     # if True, per-subject data comes from DG selection


ALGORITHMS = [
    Algorithm("erm_early_cnn",  "early",  r"\texttt{early}"),
    Algorithm("erm_ens_early_cnn", "early-ens-C",  r"\texttt{early-ens} ($C$)", dir="early_ens", n_models="C"),
    Algorithm("erm_middle_cnn", "middle", r"\texttt{middle}"),
    Algorithm("erm_middle_en_cnn", "middle-en", r"\texttt{middle-en}", dir="middle_variants"),
    Algorithm("erm_late_cnn",   "late",   r"\texttt{late}"),
    Algorithm("", "best-dg", r"\texttt{best-dg}", is_best_dg=True),
]


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def _scalar(v):
    """Unwrap single-element lists that yaml sometimes produces."""
    if isinstance(v, list):
        return v[0]
    return v


def _num_models(data):
    return str(data.get("hyperparameters", {}).get("num_models"))


def _read_metric(data, metric):
    """Across-domain (mean, std) for one metric; used to rank DG configs."""
    across = data["metrics"]["across_domain"]
    return _scalar(across[f"{metric}_mean"]), _scalar(across[f"{metric}_std"])


def _subject_indices(per_domain, base_metric):
    """Discover the held-out subject indices present for `base_metric`.

    Robust to a varying number of subjects per dataset.
    """
    pat = re.compile(rf"^(\d+)_{re.escape(base_metric)}_mean$")
    return sorted(int(m.group(1)) for k in per_domain if (m := pat.match(k)))


def _read_per_subject(data, base_metric):
    """Per-subject (mean, std) for one config, ordered by held-out subject index.

    The subject is the OOD unit: `mean` is its seed-averaged score and `std` is
    the across-seed spread, drawn as the error bar. Order is subject-index
    ascending, so array position 0 is subject 0 -- that ordering drives the
    horizontal placement. Missing `_std` falls back to 0 (no bar) rather than
    erroring, so datasets without a stored per-subject std still plot.
    """
    per = data["metrics"]["per_domain"]
    idxs = _subject_indices(per, base_metric)
    means = np.asarray(
        [_scalar(per[f"{i}_{base_metric}_mean"]) for i in idxs], dtype=float)
    stds = np.asarray(
        [_scalar(per.get(f"{i}_{base_metric}_std", 0.0)) for i in idxs], dtype=float)
    return means, stds


def _best_dg_config(dataset_key, algo_name):
    """Champion (mean, data) within one DG algorithm, ranked by SELECT_METRIC."""
    matches = sorted(glob.glob(
        str(DIRS["dg"] / f"{dataset_key}_{algo_name}_early_cnn_*.yaml")))
    if not matches:
        return None, None
    best = (None, None)  # (mean, data)
    for m in matches:
        with open(m) as f:
            data = yaml.safe_load(f)
        mean, _ = _read_metric(data, SELECT_METRIC)
        if best[0] is None or mean > best[0]:
            best = (mean, data)
    return best


def _champion_dg_data(dataset_key):
    """Two-level DG selection; returns the champion config's data dict (or None).

    Within each algorithm: best config by SELECT_METRIC. Across algorithms: the
    highest of those. Selection is on the across-domain mean; the caller reads
    whatever it wants (here, the per-subject distribution) from the winner.
    """
    winners = []  # (mean, name, data)
    for name in DG_ALGORITHMS:
        mean, data = _best_dg_config(dataset_key, name)
        if mean is not None:
            winners.append((mean, name, data))
    if not winners:
        print(f"  [warn] no dg configs for {dataset_key}")
        return None
    mean, name, data = max(winners, key=lambda t: t[0])
    print(f"  [{dataset_key}] best-dg = {name} ({SELECT_METRIC}={mean:.4f})")
    return data


def per_subject_scores(dataset_key, algo: Algorithm, base_metric):
    """Per-subject (means, stds) from the config matching {dataset}_{key}_*.yaml.

    For best-dg, the config is the DG champion for this dataset instead of a
    single named file. When algo.n_models is set, several files share the stem
    and differ only by hyperparameters.num_models inside; pick the one that
    matches. Returns two empty arrays if no config is found.
    """
    empty = (np.array([]), np.array([]))

    if algo.is_best_dg:
        data = _champion_dg_data(dataset_key)
        return _read_per_subject(data, base_metric) if data is not None else empty

    pattern = str(DIRS[algo.dir] / f"{dataset_key}_{algo.key}_*.yaml")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  [warn] no config for {dataset_key}_{algo.key}")
        return empty

    if algo.n_models is not None:
        want = str(algo.n_models)
        chosen = []
        for m in matches:
            with open(m) as f:
                data = yaml.safe_load(f)
            if _num_models(data) == want:
                chosen.append((m, data))
        if not chosen:
            print(f"  [warn] no config for {dataset_key}_{algo.key} "
                  f"with num_models={want}")
            return empty
        if len(chosen) > 1:
            print(f"  [warn] multiple configs for {dataset_key}_{algo.key} "
                  f"num_models={want}, using: {Path(chosen[-1][0]).name}")
        data = chosen[-1][1]
    else:
        if len(matches) > 1:
            print(f"  [warn] multiple configs for {dataset_key}_{algo.key}, using: "
                  f"{Path(matches[-1]).name}")
        with open(matches[-1]) as f:
            data = yaml.safe_load(f)

    return _read_per_subject(data, base_metric)


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def _subject_offsets(n, spread=SPREAD):
    """Deterministic offset per subject: subject 0 -> left, last -> right.

    Evenly spaced across [-spread/2, +spread/2]. A single subject sits centred.
    """
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-spread / 2, spread / 2, n)


def _draw_points(ax, pos, vals, stds, method):
    """One marker per subject at its subject-index offset, with a seed-std bar."""
    colour = ps.colour_for(method)
    ax.errorbar(
        pos + _subject_offsets(len(vals)), vals, yerr=stds,
        linestyle="none", marker=ps.marker_for(method), color=colour,
        ecolor=colour, elinewidth=0.8, capsize=1.5, zorder=3,
    )


def make_plot(datasets=DATASETS, algorithms=ALGORITHMS, base_metric=SUBJECT_METRIC,
              metric_label=SUBJECT_METRIC_LABEL):
    """One tall, full-width figure: a stacked row of per-dataset panels.

    Each row is one dataset. Within a method's slot, points are placed left-to-
    right by held-out subject index: subject 0 at the left edge, the last subject
    at the right edge, with an error bar showing the across-seed std for that
    subject. Because every method uses the same ordering, a point's horizontal
    position is the same subject across methods, so you can read a given subject
    vertically down its column.
    """
    w = 470 / 72.27                       # full text width
    row_h = 0.85                          # height per dataset row (inches)
    fig, axes = plt.subplots(
        nrows=len(datasets), ncols=1,
        figsize=(w, row_h * len(datasets)),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        is_last = (i == len(datasets) - 1)
        for pos, algo in enumerate(algorithms):
            vals, stds = per_subject_scores(dataset, algo, base_metric)
            if vals.size == 0:
                continue
            _draw_points(ax, pos, vals, stds, algo.method)
            iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
            print(f"  [{dataset}] {algo.method:>7}: n={vals.size} "
                  f"median={np.median(vals):.3f} worst={vals.min():.3f} IQR={iqr:.3f}")

        # dataset name on the left edge; method names only along the shared bottom axis
        ax.set_ylabel(DATASET_LABELS.get(dataset, dataset), rotation=0,
                      ha="right", va="center", labelpad=5)
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels([a.label for a in algorithms] if is_last
                           else [""] * len(algorithms))
        ax.set_xlim(-0.5, len(algorithms) - 0.5)
        ps.style_axes(ax)

    fig.text(0.5, 0.995, metric_label, rotation=0, va="top", ha="center")

    handles = [plt.Line2D([0], [0], marker=ps.marker_for(a.method),
                          color=ps.colour_for(a.method), linestyle="none")
               for a in algorithms]
    labels = [a.label for a in algorithms]
    ps.add_figure_legend(fig, handles, labels)

    # fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.98))
    fig.subplots_adjust(hspace=0.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / "ood_per_subject.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved at {file_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    make_plot()