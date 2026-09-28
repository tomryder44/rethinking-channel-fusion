import os
import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import time
from collections import defaultdict

from logging_setup import setup_logging
from train import set_seed, train_model, get_loso_splits
from utils.load import load_dataloaders


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="testing")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="dsads")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda}") if torch.cuda.is_available() else torch.device("cpu")

    hyperparameters = {"lr": 5e-3, "wd": 1e-5, "num_epochs": 500, "batch_size": 64, "num_filters": 8}
    config = {"dataset": args.dataset, "algorithm": "erm", "fusion": "late_w",
              "encoder": "cnn", "num_runs": args.num_runs, "hyperparameters": hyperparameters}

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
    temperatures = torch.logspace(-2, 1, steps=50)
    tau_list = [t.item() for t in temperatures]
    epsilon = 0.01  # uniform weight threshold

    across_domain_scores = defaultdict(list)

    for run in range(num_runs):
        print(f"on run {run + 1}/{num_runs}")
        set_seed(base_seed + run + 1)
        across_domain_scores_run = defaultdict(list)

        splits = get_loso_splits(args.dataset)

        for split in splits.values():
            target_set = split["test"]
            source_set = split["train"]

            source_train_loader, source_val_loader, target_loader = \
                load_dataloaders(dataset=args.dataset,
                                 source=source_set,
                                 target=target_set,
                                 device=device)

            training_stats, alg = train_model(source_train_loader, source_val_loader, target_loader, config, device)
            final_epoch = list(training_stats["val"].keys())[-1]
            val_stats = training_stats["val"][final_epoch]
            test_stats = training_stats["test"][final_epoch]

            # uniform baseline
            across_domain_scores_run["uniform"].append(float(test_stats["macro_f1"]))

            # per-channel val F1s (for building weights)
            channel_val_f1s = torch.tensor(
                [val_stats[k] for k in sorted((k for k in val_stats if k.endswith("_macro_f1")),
                                              key=lambda k: int(k.split("_")[0]))]
            )

            # tau sweep
            tau_close = None
            for tau in tau_list:
                new_weights = F.softmax(channel_val_f1s / tau, dim=0).to(device)
                out = alg.get_features(target_loader, weights=new_weights)
                f1 = float(alg.metric_computer.compute(out)["macro_f1"])
                across_domain_scores_run[f"tau_{tau:.6g}"].append(f1)

                uniform_w = 1.0 / channel_val_f1s.numel()
                if tau_close is None and float((new_weights - uniform_w).abs().max()) < epsilon:
                    tau_close = tau
            across_domain_scores_run["tau_star"].append(tau_close if tau_close is not None else tau_list[-1])

        # average over domains for this run
        for key, vals in across_domain_scores_run.items():
            across_domain_scores[f"{key}_mean"].append(float(np.mean(vals)))
            across_domain_scores[f"{key}_worst"].append(float(np.min(vals)))

    # average + std across seeds
    across_domain_final = {}
    for key, vals in across_domain_scores.items():
        across_domain_final[f"{key}_mean"] = float(np.mean(vals))
        across_domain_final[f"{key}_std"] = float(np.std(vals))

    config["results"] = {"across_domain": across_domain_final}

    with open(config_file_path, "w") as f:
        yaml.safe_dump(config, f)
