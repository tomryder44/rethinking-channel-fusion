import glob
import yaml
from pathlib import Path
from typing import NamedTuple


DIRS = {
    "main":            Path("outputs/configs/main"),
    "dg":              Path("outputs/configs/dg"),
    "early_ens":       Path("outputs/configs/early_ens"),
    "filters":         Path("outputs/configs/filters"),
    "middle_variants": Path("outputs/configs/middle_variants"),
}

SELECT_METRIC = "test_macro_f1_mean"
DG_ALGORITHMS = ["diversify", "groupdro", "irm", "mmd", "nnr", "phaser", "vrex"]


class Algorithm(NamedTuple):
    key: str
    label: str
    dir: str = "main"
    select: tuple[str, str] | None = None
    is_best_dg: bool = False
    is_dg_swept: bool = False


def latex_escape(s):
    repl = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    return ''.join(repl.get(c, c) for c in s)


def _scalar(v):
    return v[0] if isinstance(v, list) else v


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _read_metric(data, metric):
    across = data["metrics"]["across_domain"]
    return _scalar(across[f"{metric}_mean"]), _scalar(across[f"{metric}_std"])


def _hyperparam(data, name):
    return str(data.get("hyperparameters", {}).get(name))


def _best_config(dir_key, key, dataset_key):
    matches = sorted(glob.glob(str(DIRS[dir_key] / f"{dataset_key}_{key}_*.yaml")))
    if not matches:
        print(f"  [warn] no config for {dataset_key}_{key}")
        return None, None, None
    best = (None, None, None)
    for m in matches:
        data = _load(m)
        mean, _ = _read_metric(data, SELECT_METRIC)
        if best[0] is None or mean > best[0]:
            best = (mean, m, data)
    return best


def load_metric(dataset_key, algo, metric):
    matches = sorted(glob.glob(str(DIRS[algo.dir] / f"{dataset_key}_{algo.key}_*.yaml")))
    if not matches:
        print(f"  [warn] no config for {dataset_key}_{algo.key}")
        return None, None

    if algo.select is not None:
        name, want = algo.select
        want = str(want)
        data = None
        for m in matches:
            d = _load(m)
            if _hyperparam(d, name) == want:
                data = d
        if data is None:
            print(f"  [warn] no config for {dataset_key}_{algo.key} with {name}={want}")
            return None, None
    else:
        data = _load(matches[-1])

    return _read_metric(data, metric)


def select_best_config(dataset_key, algo, metric):
    mean, path, data = _best_config(algo.dir, algo.key, dataset_key)
    if mean is None:
        return None, None
    print(f"  [{dataset_key}] {algo.key}: {Path(path).name} ({SELECT_METRIC}={mean:.4f})")
    return _read_metric(data, metric)


def select_best_dg(dataset_key, metric):
    winners = []
    for name in DG_ALGORITHMS:
        mean, path, data = _best_config("dg", f"{name}_early_cnn", dataset_key)
        if mean is not None:
            winners.append((mean, name, path, data))
    if not winners:
        print(f"  [{dataset_key}] best-dg: none found")
        return None, None
    mean, name, path, data = max(winners, key=lambda t: t[0])
    print(f"  [{dataset_key}] best-dg = {name} @ {Path(path).name} ({SELECT_METRIC}={mean:.4f})")
    return _read_metric(data, metric)


def load_results(algorithms, datasets, metric):
    results, results_std = {}, {}
    for algo in algorithms:
        for ds_key in datasets:
            if algo.is_best_dg:
                mean, std = select_best_dg(ds_key, metric)
            elif algo.is_dg_swept:
                mean, std = select_best_config(ds_key, algo, metric)
            else:
                mean, std = load_metric(ds_key, algo, metric)
            results[(algo, ds_key)] = mean if mean is not None else 0.0
            results_std[(algo, ds_key)] = std if std is not None else 0.0
    return results, results_std


def make_latex_table(datasets, algorithms, results, results_std,
                     decimals=3, std_decimals=None, missing_str=r"--", bold="max"):
    if std_decimals is None:
        std_decimals = decimals
    tol = 1e-12
    bold_mode = (bold or "").strip().lower()

    col_best = {}
    for ds_key in datasets:
        vals = [results.get((algo, ds_key)) for algo in algorithms]
        vals = [v for v in vals if v is not None and abs(v) >= tol]
        if not vals:
            col_best[ds_key] = None
        else:
            col_best[ds_key] = min(vals) if bold_mode == "min" else max(vals)

    rows = []
    for algo in algorithms:
        row = [algo.label]
        for ds_key in datasets:
            v = results.get((algo, ds_key))
            if v is None:
                cell = missing_str
            elif abs(v) < tol:
                cell = "/"
            else:
                mean_str = f"{v:.{decimals}f}"
                best = col_best.get(ds_key)
                if best is not None and abs(v - best) <= tol and bold_mode in ("min", "max"):
                    mean_str = rf"\textbf{{{mean_str}}}"
                s = results_std.get((algo, ds_key))
                if s is not None:
                    cell = rf"{mean_str}{{\tiny$\pm${s:.{std_decimals}f}}}"
                else:
                    cell = mean_str
            row.append(cell)
        rows.append("      " + " & ".join(row) + r" \\")
    return rows


def print_combined_table(datasets, algorithms, blocks):
    n_data_cols = len(datasets)
    n_total_cols = 1 + n_data_cols
    col_spec = "l" + "c" * n_data_cols

    header_cells = [" "] + [f"\\texttt{{{latex_escape(datasets[ds])}}}" for ds in datasets]

    lines = [f"\\begin{{tabular}}{{{col_spec}}}", r"\toprule",
             "      " + " & ".join(header_cells) + r" \\", r"\midrule"]

    for i, block in enumerate(blocks):
        results, results_std = load_results(algorithms, datasets, block["metric"])
        arrow = r"$\uparrow$" if block.get("bold") == "max" else r"$\downarrow$"
        block_header = f"\\textbf{{{block['title']}}} ({block['label']}) ({arrow})"
        lines.append(f"\\rowcolor{{gray!15}} \\multicolumn{{{n_total_cols}}}{{l}}{{{block_header}}} \\\\")
        lines.extend(make_latex_table(datasets, algorithms, results, results_std,
                                      decimals=block.get("decimals", 3),
                                      bold=block.get("bold", "max")))
        if i < len(blocks) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}"]
    print("\n".join(lines))


if __name__ == "__main__":
    datasets = {
        "dsads": "DSADS", "mhealth": "MHEALTH", "pamap": "PAMAP", "wisdm": "WISDM",
        "bnci1": "BNCI-1", "bnci2": "BNCI-2", "bnci4": "BNCI-4", "zhou": "ZHOU",
    }

    algorithms = [
        Algorithm("erm_early_cnn", r"\texttt{early}"),
        Algorithm("erm_middle_cnn", r"\texttt{middle}"),
        Algorithm("erm_late_cnn", r"\texttt{late}"),
        Algorithm("erm_ens_early_cnn", r"\texttt{early-ens} ($5$)", dir="early_ens", select=("num_models", "5")),
        Algorithm("erm_ens_early_cnn", r"\texttt{early-ens} ($10$)", dir="early_ens", select=("num_models", "10")),
        Algorithm("erm_ens_early_cnn", r"\texttt{early-ens} ($C$)", dir="early_ens", select=("num_models", "C")),
        Algorithm("erm_early_cnn", r"\texttt{early} ($16$)", dir="filters", select=("num_filters", "16")),
        Algorithm("erm_early_cnn", r"\texttt{early} ($32$)", dir="filters", select=("num_filters", "32")),
        Algorithm("erm_early_cnn", r"\texttt{early} ($64$)", dir="filters", select=("num_filters", "64")),
        Algorithm("erm_middle_e_cnn", r"\texttt{middle-e}", dir="middle_variants"),
        Algorithm("erm_middle_n_cnn", r"\texttt{middle-n}", dir="middle_variants"),
        Algorithm("erm_middle_en_cnn", r"\texttt{middle-en}", dir="middle_variants"),
        Algorithm("diversify_early_cnn", r"\texttt{DIVERSIFY}", dir="dg", is_dg_swept=True),
        Algorithm("groupdro_early_cnn", r"\texttt{GroupDRO}", dir="dg", is_dg_swept=True),
        Algorithm("irm_early_cnn", r"\texttt{IRM}", dir="dg", is_dg_swept=True),
        Algorithm("mmd_early_cnn", r"\texttt{MMD}", dir="dg", is_dg_swept=True),
        Algorithm("nnr_early_cnn", r"\texttt{NNR}", dir="dg", is_dg_swept=True),
        Algorithm("phaser_early_cnn", r"\texttt{PhASER}", dir="dg", is_dg_swept=True),
        Algorithm("vrex_early_cnn", r"\texttt{VREx}", dir="dg", is_dg_swept=True),
        Algorithm("", r"\texttt{best-dg}", is_best_dg=True),
    ]

    print_combined_table(datasets, algorithms, blocks=[
        {"label": r"$m_{\text{avg}}^{\text{ID}}$", "title": "Average",
         "metric": "val_macro_f1_mean", "bold": "max", "decimals": 3},
    ])

    print_combined_table(datasets, algorithms, blocks=[
        {"label": r"$m_{\text{avg}}^{\text{OOD}}$", "title": "Average",
         "metric": "test_macro_f1_mean", "bold": "max", "decimals": 3},
        {"label": r"$m_{\text{worst}}^{\text{OOD}}$", "title": "Worst-group",
         "metric": "test_macro_f1_min", "bold": "max", "decimals": 3},
    ])