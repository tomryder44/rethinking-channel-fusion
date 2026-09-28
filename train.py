import random
import numpy as np
import torch
import yaml
import time
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

from utils.split import get_loso_splits
from utils.load import load_dataloaders
from mapping import get_classes


def set_seed(seed):
    logger.debug(f"SEED: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(source_train_loader, source_val_loader, target_loader, config, device):

    num_channels = source_train_loader.dataset.num_channels
    num_classes = source_train_loader.dataset.num_classes
    logger.debug(f"number of classes: {num_classes}")
    logger.debug(f"number of channels: {num_channels}")

    encoder_class, model_class, trainer_class = get_classes(config["encoder"], config["fusion"], config["algorithm"])
    logger.debug(f"encoder: {encoder_class}")
    logger.debug(f"model: {model_class}")
    logger.debug(f"trainer: {trainer_class}")

    model = model_class(encoder_class, num_classes, num_channels, config["hyperparameters"]).to(device)
    alg = trainer_class(model, num_classes, num_channels, device, config["hyperparameters"])

    logger.debug(f"total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.debug(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    start = time.time()
    training_stats = alg.train(source_train_loader, source_val_loader, target_loader)
    logger.debug(f"train time: {time.time() - start:.1f} seconds")

    return training_stats, alg


def run_experiment(config_file_path: str, device):

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"config: {config}")

    num_runs = config["num_runs"]
    base_seed = 42069

    per_domain_scores = defaultdict(list)
    across_domain_scores = defaultdict(list)

    for run in range(num_runs):
        print(f"on run {run+1}/{num_runs}")
        set_seed(base_seed + run + 1)
        across_domain_scores_run = defaultdict(list)

        splits = get_loso_splits(config["dataset"])

        for split in splits.values():
            target_set = split["test"]
            source_set = split["train"]
            logger.debug(f"TARGET DOMAIN: {target_set}")

            source_train_loader, source_val_loader, target_loader = \
                load_dataloaders(dataset=config["dataset"],
                                 source=source_set,
                                 target=target_set,
                                 device=device,
                                 algorithm=config["algorithm"],
                                 nperseg_k=config["hyperparameters"].get("nperseg_k"))

            training_stats, alg = train_model(source_train_loader, source_val_loader, target_loader, config, device)

            if config["algorithm"] == "erm" and config["encoder"] == "cnn" and config["fusion"] in ("early", "middle"):
                alg.measure_domain_divergence(source_val_loader, target_loader)
                training_stats = alg.training_stats

            target = target_set[0]
            final_epoch = list(training_stats["val"].keys())[-1]

            for phase, phase_dict in training_stats.items():  # train, val, test
                for metric, value in phase_dict[final_epoch].items(): # loss, macro_f1
                    val = float(value)
                    per_domain_scores[f"{target}_{phase}_{metric}"].append(val)  # 0_test_loss
                    across_domain_scores_run[f"{phase}_{metric}"].append(val)  # test_loss

        # compute across domain mean/min/max for each metric
        for metric_name, vals in across_domain_scores_run.items():
            vals = np.asarray(vals, dtype=float)
            across_domain_scores[f"{metric_name}_mean"].append(float(np.mean(vals)))
            across_domain_scores[f"{metric_name}_min"].append(float(np.min(vals)))
            across_domain_scores[f"{metric_name}_max"].append(float(np.max(vals)))

    # add average and std across experiment runs
    for metric, vals in list(per_domain_scores.items()):
        per_domain_scores[f"{metric}_mean"] = float(np.mean(vals))
        per_domain_scores[f"{metric}_std"] = float(np.std(vals))

    for metric, vals in list(across_domain_scores.items()):
        across_domain_scores[f"{metric}_mean"] = float(np.mean(vals))
        across_domain_scores[f"{metric}_std"] = float(np.std(vals))

    config["metrics"] = {"per_domain": dict(per_domain_scores), "across_domain": dict(across_domain_scores)}

    with open(config_file_path, "w") as f:
        yaml.safe_dump(config, f)
