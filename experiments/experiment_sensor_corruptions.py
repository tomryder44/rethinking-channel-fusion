import os
import argparse
import torch
import yaml
import numpy as np
from itertools import combinations
import time

from logging_setup import setup_logging
from train import set_seed, train_model
from utils.split import get_loso_splits
from utils.load import load_dataloaders


SENSOR_GROUPS = {
    "dsads":  [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14]
    ],
    "mhealth": [
        [0, 1, 2],  # chest accelerometer
        [5, 6, 7, 8, 9, 10, 11, 12, 13],  # left-ankle IMU
        [14, 15, 16, 17, 18, 19, 20, 21, 22],  # right-lower-arm IMU
    ],
    "pamap": [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # hand IMU
        [11, 12, 13, 14, 15, 16, 17, 18, 19],  # chest IMU
        [21, 22, 23, 24, 25, 26, 27, 28, 29],  # ankle IMU
    ],
}


HAR = ("dsads", "mhealth", "pamap")


def apply_shifts(x, channels, shift):
    if shift == "zero":
        x[:, channels, :] = 0.
    elif shift == "noise":
        var = torch.empty(1, device=x.device).uniform_(1, 3.0)
        x[:, channels, :] += torch.randn_like(x[:, channels, :]) * var
    elif shift == "saturate":
        sat = torch.empty(1, device=x.device).uniform_(1.0, 3.0)
        x[:, channels, :] = sat
    return x


def analyse_corrupt(alg, loader, channels, shift):
    preds = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            x = batch["sequence"]
            labels.append(batch["label"].detach())
            x = apply_shifts(x, channels, shift)
            out = alg.model(x)
            try:
                preds.append(out["logits"].detach())
            except:  # for late fusion
                preds.append(out["probs"].detach())
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        mf1 = alg.metric_computer.compute_macro_f1(preds, labels)
    return mf1


def analyse_corruptions(alg, loader, metrics, fusion, max_k, num_draws=10):
    for s in ["zero", "saturate", "noise"]:
        for k in range(0, max_k + 1):
            draws = [torch.randperm(num_channels)[:k] for _ in range(num_draws)]
            draw_scores = [analyse_corrupt(alg, loader, channels=channels, shift=s) for channels in draws]
            mean_score = float(np.mean([float(v) for v in draw_scores]))
            metrics[fusion][s].setdefault(k, []).append(mean_score)


def sensor_draws(groups, n_fail, max_draws=10):
    combos = list(combinations(range(len(groups)), n_fail))
    if len(combos) > max_draws:
        idx = torch.randperm(len(combos))[:max_draws].tolist()
        combos = [combos[i] for i in idx]
    return [sorted(c for g in combo for c in groups[g]) for combo in combos]


def analyse_sensor_corruptions(alg, loader, metrics, fusion, groups):
    n_sensors = len(groups)
    for s in ["zero", "saturate", "noise"]:
        for n_fail in range(0, n_sensors):
            draws = sensor_draws(groups, n_fail)
            draw_scores = [analyse_corrupt(alg, loader, channels=channels, shift=s) for channels in draws]
            mean_score = float(np.mean([float(v) for v in draw_scores]))
            metrics[fusion][s].setdefault(n_fail, []).append(mean_score)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="testing")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="dsads")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda}") if torch.cuda.is_available() else torch.device("cpu")

    hyperparameters = {"lr": 5e-3, "wd": 1e-5, "num_epochs": 500, "batch_size": 64, "num_filters": 8,
                       "num_models": "C"  # for the early fusion ensemble only
                       }
    config = {"dataset": args.dataset, "encoder": "cnn", "num_runs": args.num_runs, "hyperparameters": hyperparameters}

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

    num_runs = args.num_runs
    base_seed = 42069

    fusions = [
        ("early", "erm"),
        ("early", "erm_ens"),
        ("middle", "erm"),
        ("middle_en", "erm"),
        ("late", "erm"),
    ]

    def fusion_key(fusion, algorithm):
        return fusion if algorithm == "erm" else f"{fusion}_{algorithm}"
    metrics = {fusion_key(f, a): {s: {} for s in ["zero", "saturate", "noise"]} for f, a in fusions}
    if args.dataset in HAR:
        sensor_metrics = {fusion_key(f, a): {s: {} for s in ["zero", "saturate", "noise"]} for f, a in fusions}

    for run in range(num_runs):
        print(f"on run {run+1}/{num_runs}")
        set_seed(base_seed + run + 1)

        splits = get_loso_splits(args.dataset)
        split = splits["split_0"]
        target_set = split["test"]
        source_set = split["train"]

        source_train_loader, source_val_loader, target_loader = load_dataloaders(args.dataset, source_set, target_set, device)
        num_channels = source_val_loader.dataset.num_channels
        max_k = num_channels // 2

        for f, a in fusions:
            config["fusion"] = f
            config["algorithm"] = a
            key = fusion_key(f, a)
            _, alg = train_model(source_train_loader, source_val_loader, target_loader, config, device)
            analyse_corruptions(alg=alg, loader=source_val_loader, metrics=metrics, fusion=key, max_k=max_k)
            if args.dataset in HAR:
                groups = SENSOR_GROUPS[args.dataset]
                analyse_sensor_corruptions(alg=alg, loader=source_val_loader, metrics=sensor_metrics, fusion=key, groups=groups)

    config["results"] = {"channel": metrics}
    if args.dataset in HAR:
        config["results"]["sensor"] = sensor_metrics

    with open(config_file_path, "w") as f:
        yaml.safe_dump(config, f)
