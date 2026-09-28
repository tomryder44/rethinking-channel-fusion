import argparse
import torch
import time
import numpy as np

from mapping import get_classes
from train import get_loso_splits
from utils.load import load_dataloaders

torch.backends.cudnn.benchmark = False


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


dataset_display = {
    "wisdm": r"\texttt{WISDM}",
    "dsads": r"\texttt{DSADS}",
    "mhealth": r"\texttt{MHEALTH}",
    "pamap": r"\texttt{PAMAP}",
}

encoder_display = {
    "cnn": "CNN",
    "cnn_lstm": "CNN-LSTM",
    "transformer": "transformer",
}

fusion_display = {
    "early": r"\texttt{early}",
    "middle": r"\texttt{middle}",
    "late": r"\texttt{late}",
}

channel_counts = {"wisdm": 3, "dsads": 15, "mhealth": 23, "pamap": 30}


def time_model(alg, loader, device, num_warmup, num_timed):
    """Return (mean train ms, mean infer ms) per batch, skipping warmup epochs."""
    num_epochs = num_warmup + num_timed

    train_times = []
    alg.model.train()
    for epoch in range(num_epochs):
        for batch in loader:
            sync(device)
            t0 = time.perf_counter()
            alg.update(batch)
            sync(device)
            if epoch >= num_warmup:
                train_times.append(time.perf_counter() - t0)

    infer_times = []
    alg.model.eval()
    with torch.no_grad():
        for epoch in range(num_epochs):
            for batch in loader:
                sync(device)
                t0 = time.perf_counter()
                alg.model(batch["sequence"])
                sync(device)
                if epoch >= num_warmup:
                    infer_times.append(time.perf_counter() - t0)

    return np.mean(train_times) * 1e3, np.mean(infer_times) * 1e3


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="2")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--timed_epochs", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}") if torch.cuda.is_available() else torch.device("cpu")

    datasets = ["wisdm", "dsads", "mhealth", "pamap"]

    rows = [
        ("cnn", "early", "erm"),
        ("cnn", "middle", "erm"),
        ("cnn", "late", "erm"),
        ("cnn_lstm", "early", "erm"),
        ("cnn_lstm", "middle", "erm"),
        ("cnn_lstm", "late", "erm"),
        ("transformer", "early", "erm"),
        ("transformer", "middle", "erm"),
        ("transformer", "late", "erm"),
    ]

    hyperparameters = {"lr": 5e-3, "wd": 1e-5, "batch_size": 64, "num_filters": 8}

    # results[row][dataset] = (mean train ms, mean infer ms), averaged over repeats
    results = {row: {} for row in rows}

    for dataset in datasets:
        print(f"dataset: {dataset}")

        split = get_loso_splits(dataset)["split_0"]
        loader, _, _ = load_dataloaders(
            dataset=dataset, source=split["train"], target=split["test"], device=device
        )

        num_channels = loader.dataset.num_channels
        num_classes = loader.dataset.num_classes

        for row in rows:
            print(row)
            encoder, fusion, algorithm = row
            encoder_class, model_class, trainer_class = get_classes(encoder, fusion, algorithm)

            train_runs, infer_runs = [], []
            for _ in range(args.repeats):
                model = model_class(encoder_class, num_classes, num_channels, hyperparameters).to(device)
                alg = trainer_class(model, num_classes, num_channels, device, hyperparameters)
                train_ms, infer_ms = time_model(
                    alg, loader, device, args.warmup_epochs, args.timed_epochs
                )
                train_runs.append(train_ms)
                infer_runs.append(infer_ms)

            results[row][dataset] = (np.mean(train_runs), np.mean(infer_runs))

    # ---- build LaTeX table (relative to CNN-early-erm per dataset) ----

    baseline_row = ("cnn", "early", "erm")

    def cell(row, dataset):
        train_ms, infer_ms = results[row][dataset]
        base_train_ms, base_infer_ms = results[baseline_row][dataset]
        return f"{train_ms / base_train_ms:.1f}/{infer_ms / base_infer_ms:.1f}"

    header_cols = " & ".join(
        f"{dataset_display[d]}($C$={channel_counts[d]})" for d in datasets
    )

    lines = [
        r"\begin{tabular}{l" + "r" * (1 + len(datasets)) + "}",
        r"\toprule",
        r"& Encoder & " + header_cols + r" \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        encoder, fusion, algorithm = row
        cells = " & ".join(cell(row, d) for d in datasets)
        lines.append(f"{fusion_display[fusion]} & {encoder_display[encoder]} & {cells} \\\\")
        if idx == 2 or idx == 5:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]

    print("\n" + "\n".join(lines) + "\n")