import os
import argparse
import numpy as np

from paths import PROCESSED_DIR
from data_processing.preprocessing.get_dataset_subjects import get_subjects
from data_processing.loading.dsads import load_dsads
from data_processing.loading.mhealth import load_mhealth
from data_processing.loading.pamap import load_pamap
from data_processing.loading.wisdm import load_wisdm
from data_processing.loading.bnci001 import load_bnci001
from data_processing.loading.bnci002 import load_bnci002
from data_processing.loading.bnci004 import load_bnci004
from data_processing.loading.zhou import load_zhou


load_dict = {
    "dsads": load_dsads,
    "mhealth": load_mhealth,
    "pamap": load_pamap,
    "wisdm": load_wisdm,
    "bnci1": load_bnci001,
    "bnci2": load_bnci002,
    "bnci4": load_bnci004,
    "zhou": load_zhou
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=load_dict.keys())
    args = parser.parse_args()

    dataset = args.dataset
    subjects = get_subjects(dataset)
    load_func = load_dict[dataset]

    folder_path = os.path.join(PROCESSED_DIR, dataset)
    os.makedirs(folder_path, exist_ok=True)

    for subject in subjects:
        print(f"subject {subject}")

        data, labels = load_func(subject)
        print(data.shape)

        x_file_path = os.path.join(folder_path, f"{subjects.index(subject)}_x.npy")
        y_file_path = os.path.join(folder_path, f"{subjects.index(subject)}_y.npy")

        np.save(x_file_path, data)
        np.save(y_file_path, labels)
