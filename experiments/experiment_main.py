import os
import argparse
import torch
import yaml
import time

from logging_setup import setup_logging
from train import run_experiment


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="testing")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="dsads")
    parser.add_argument("--algorithm", type=str, default="erm")
    parser.add_argument("--fusion", type=str, default="early")
    parser.add_argument("--encoder", type=str, default="cnn")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")  # e.g. --set penalty=0.1 --set num_filters=16")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda}") if torch.cuda.is_available() else torch.device("cpu")

    hyperparameters = {"lr": 5e-3, "wd": 1e-5, "num_epochs": 500, "batch_size": 64, "num_filters": 8}
    for override in args.set:
        key, _, value = override.partition("=")
        hyperparameters[key] = yaml.safe_load(value)

    config = {"dataset": args.dataset, "algorithm": args.algorithm, "fusion": args.fusion,
              "encoder": args.encoder, "num_runs": args.num_runs, "hyperparameters": hyperparameters}

    print(f"using device: {device}")
    print(f"config: {config}")

    config_folder_path = os.path.join("outputs/configs", args.exp_name)
    os.makedirs(config_folder_path, exist_ok=True)

    run_id = f"{args.dataset}_{args.algorithm}_{args.fusion}_{args.encoder}_{int(time.time() * 1000)}"
    config_file_path = os.path.join(config_folder_path, f"{run_id}.yaml")
    print(f"config file path: {config_file_path}")

    with open(config_file_path, "w") as f:
        yaml.safe_dump(config, f)

    log_file_path = config_file_path.replace("configs", "logs").replace(".yaml", ".log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    setup_logging(log_file_path)

    run_experiment(config_file_path, device)