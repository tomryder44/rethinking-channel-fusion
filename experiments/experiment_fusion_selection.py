import os
import argparse
import torch
import yaml
import numpy as np
import time
from collections import defaultdict

from logging_setup import setup_logging
from train import set_seed, train_model
from utils.split import get_loso_splits
from utils.load import load_dataloaders


FUSIONS = ["early", "middle", "late"]


def fit(dataset, source, target, fusion, config, device):
    config = dict(config)
    config["fusion"] = fusion
    train_loader, val_loader, target_loader = load_dataloaders(
        dataset=dataset, source=source, target=target, device=device)
    stats, _ = train_model(train_loader, val_loader, target_loader, config, device)
    final_epoch = list(stats["val"].keys())[-1]
    return float(stats["test"][final_epoch]["macro_f1"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="nested")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mhealth")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda}") if torch.cuda.is_available() else torch.device("cpu")

    hyperparameters = {"lr": 5e-3, "wd": 1e-5, "num_epochs": 500, "batch_size": 64, "num_filters": 8}
    config = {"dataset": args.dataset, "algorithm": "erm", "num_runs": args.num_runs, "encoder": "cnn",
              "hyperparameters": hyperparameters}

    print(f"using device: {device}")
    print(f"config: {config}")

    config_folder_path = os.path.join("outputs/configs", args.exp_name)
    os.makedirs(config_folder_path, exist_ok=True)

    run_id = f"{args.dataset}_{int(time.time() * 1000)}"
    config_file_path = os.path.join(config_folder_path, f"{run_id}.yaml")
    print(f"config file path: {config_file_path}")

    with open(config_file_path, "w") as f:
        yaml.safe_dump(config, f)

    log_file_path = config_file_path.replace("configs", "logs").replace(".yaml", ".log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    setup_logging(log_file_path)

    # ~~~

    dataset = config["dataset"]
    num_runs = config["num_runs"]
    base_seed = 42069

    domains = sorted({s for split in get_loso_splits(dataset).values() for s in split["test"]})

    # one entry per seed (already reduced across domains) -- averaged across seeds at the end
    across_domain_scores = defaultdict(list)

    for run in range(num_runs):
        print(f"on run {run + 1}/{num_runs}")
        set_seed(base_seed + run + 1)

        # --- inner loop: for each (d_i, d'), train all fusions on E \ {d_i, d'} ---
        max_dom = 15
        sub_dom = 5
        rng = np.random.default_rng(base_seed + run + 1)
        subsample = len(domains) > max_dom
        inner = {}
        for d_i in domains:
            inner[d_i] = {}
            others = [d for d in domains if d != d_i]
            if subsample:
                d_primes = rng.choice(others, size=sub_dom, replace=False).tolist()
            else:
                d_primes = others
            for d_prime in d_primes:
                source = [d for d in domains if d not in (d_i, d_prime)]
                inner[d_i][d_prime] = {f: fit(dataset, source, [d_prime], f, config, device) for f in FUSIONS}

        # --- outer: train all fusions on E \ {d_i}, evaluate on d_i ---
        outer = {}
        for d_i in domains:
            source = [d for d in domains if d != d_i]
            outer[d_i] = {f: fit(dataset, source, [d_i], f, config, device) for f in FUSIONS}

        # --- selection: per domain, pick fusion with best mean inner OOD test over held-out d' ---
        chosen_scores = []
        oracle_scores = []
        fixed_scores = {f: [] for f in FUSIONS}
        chosen_fusions = []  # which fusion was picked per domain
        for d_i in domains:
            inner_test_mean = {
                f: float(np.mean([inner[d_i][d_prime][f] for d_prime in inner[d_i]])) for f in FUSIONS}
            chosen = max(inner_test_mean, key=inner_test_mean.get)
            chosen_fusions.append(chosen)
            chosen_scores.append(outer[d_i][chosen])
            oracle_scores.append(max(outer[d_i][f] for f in FUSIONS))
            for f in FUSIONS:
                fixed_scores[f].append(outer[d_i][f])

        # --- reduce across domains for this seed: mean + worst-group (min) ---
        for name, vals in [("nested", chosen_scores), ("oracle", oracle_scores), *[(f, fixed_scores[f]) for f in FUSIONS]]:
            vals = np.asarray(vals, dtype=float)
            across_domain_scores[f"{name}_mean"].append(float(np.mean(vals)))
            across_domain_scores[f"{name}_worst"].append(float(np.min(vals)))

        # selection proportion per fusion, this seed
        for f in FUSIONS:
            prop = chosen_fusions.count(f) / len(chosen_fusions)
            across_domain_scores[f"prop_{f}"].append(prop)

    # --- average + std across seeds ---
    results = {}
    for key, vals in across_domain_scores.items():
        results[f"{key}_mean"] = float(np.mean(vals))
        results[f"{key}_std"] = float(np.std(vals))

    # --- summary ---
    print("\n=== summary (across seeds) ===")
    print(f"{'':>16} {'mean':>7} {'worst':>7}")
    for name in ["nested", "oracle", *FUSIONS]:
        label = name if name in ("nested", "oracle") else f"{name} (fixed)"
        print(f"{label:>16} {results[f'{name}_mean_mean']:>7.3f} {results[f'{name}_worst_mean']:>7.3f}")
    print(f"\nregret (mean):  {results['oracle_mean_mean'] - results['nested_mean_mean']:.4f}")

    config["results"] = results
    with open(config_file_path, "w") as f:
        yaml.safe_dump(config, f)
