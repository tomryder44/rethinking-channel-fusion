import os
import numpy as np
import pandas as pd

from paths import RAW_DIR


def load_wisdm(subject: str):

    dataset_path = os.path.join(RAW_DIR, "wisdm", "WISDM_ar_v1.1_raw.txt")

    # there are some rows that have extra columns that cause a parser error
    columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    all_data = pd.read_csv(dataset_path,
                           header=None,
                           names=columns,
                           dtype={'user': int, 'activity': object, 'timestamp': int, 'x-axis': float, 'y-axis': float,
                                  'z-axis': float, 'a1': object, 'a2': float, 'a3': float, 'a4': float, 'a5': float, 'a6': float})
    all_data.drop(['timestamp', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'], axis=1, inplace=True)

    # remove rows with missing values
    all_data.dropna(inplace=True)

    # classes are given as activity labels
    class_mapping = {"Walking": 0,
                     "Jogging": 1,
                     "Sitting": 2,
                     "Standing": 3,
                     "Upstairs": 4,
                     "Downstairs": 5}
    all_data.replace({'activity': class_mapping}, inplace=True)

    # unique subject indices are from '1' to '36'

    subject_data = all_data[all_data['user'] == int(subject)].copy()
    subject_data.drop(['user'], axis=1, inplace=True)

    subject_y = subject_data['activity']
    subject_x = subject_data.drop(['activity'], axis=1)

    subject_x = subject_x.to_numpy()
    subject_y = subject_y.to_numpy()

    window_length = 128
    data = []
    labels = []
    for idx in range(window_length, subject_x.shape[0], window_length):
        window = np.transpose(subject_x[idx - window_length:idx, :])
        first_label = subject_y[idx-window_length]
        last_label = subject_y[idx-1]
        if first_label == last_label:
            data.append(window)
            labels.append(first_label)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels



