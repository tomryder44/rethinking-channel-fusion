import os
import pandas as pd
import numpy as np

from paths import RAW_DIR


def get_subjects(dataset: str):

    dataset_path = RAW_DIR

    if dataset == "dsads":
        path = os.path.join(dataset_path, "dsads/a01")
        subjects = sorted(os.listdir(path))

    elif dataset == "mhealth":
        path = os.path.join(dataset_path, "mhealth")
        subjects = sorted(os.listdir(path))
        subjects = [subject.split(".")[0] for subject in subjects]

    elif dataset == "pamap":
        path = os.path.join(dataset_path, "pamap")
        subjects = sorted(os.listdir(path))
        subjects = [subject.split(".")[0] for subject in subjects]
        subjects.remove("subject109")

    elif dataset == "wisdm":
        path = os.path.join(dataset_path, "wisdm/WISDM_ar_v1.1_raw.txt")
        columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
        data = pd.read_csv(path, header=None, names=columns,
                           dtype={'user': int, 'activity': object, 'timestamp': int, 'x-axis': float, 'y-axis': float,
                                  'z-axis': float, 'a1': object, 'a2': float, 'a3': float, 'a4': float, 'a5': float,
                                  'a6': float})
        subjects = np.unique(data['user'].to_numpy())
        subjects = [str(subject) for subject in subjects]

    elif dataset == "bnci1" or dataset == "bnci4":
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    elif dataset == "bnci2":
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]

    elif dataset == "zhou":
        return ["1", "2", "3", "4"]

    return subjects