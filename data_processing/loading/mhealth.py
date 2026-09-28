import os
import pandas as pd
import numpy as np

from paths import RAW_DIR


def load_mhealth(subject: str):

    dataset_path = os.path.join(RAW_DIR, "mhealth")
    subject_path = os.path.join(dataset_path, f"{subject}.log")

    data = pd.read_csv(subject_path, header=None, sep='\t')

    data = data[data.iloc[:, -1] != 0]  # final column is class, class 0 is null class

    x = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    window_seconds = 2
    fs = 50
    window_length = int(window_seconds * fs)
    shift_length = window_length

    data = []
    labels = []

    for idx in range(window_length, x.shape[0], shift_length):
        first_label = y[idx - window_length]
        last_label = y[idx - 1]
        if first_label == last_label:
            window = np.transpose(x[idx - window_length:idx, :])
            data.append(window)
            labels.append(first_label-1)  # -1 because classes start at 1

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


def load_mhealth_1(subject: str):
    data, labels = load_mhealth(subject)
    return data[:, 0:3, :], labels


def load_mhealth_2(subject: str):
    data, labels = load_mhealth(subject)
    return data[:, 5:14, :], labels


def load_mhealth_3(subject: str):
    data, labels = load_mhealth(subject)
    return data[:, 14:, :], labels

