import os
import pandas as pd
import numpy as np

from paths import RAW_DIR


label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 12: 7, 13: 8, 16: 9, 17: 10, 24: 11}


def load_pamap(subject: str):

    dataset_path = os.path.join(RAW_DIR, "pamap")
    subject_path = os.path.join(dataset_path, f"{subject}.dat")

    data = pd.read_table(subject_path, header=None, sep='\s+')

    #   1.		timestamp (s)
    #   2.		activityID (see below for the mapping to the activities)
    #   3.		heart rate (bpm)
    #   4-20.	IMU hand
    #   21-37.	IMU chest
    #   38-54.	IMU ankle

    # The IMU sensory data contains the following columns:
    #   1.		temperature (Â°C)
    #   2-4.	3D-acceleration data (ms-2), scale: Â±16g, resolution: 13-bit
    #   5-7.	3D-acceleration data (ms-2), scale: Â±6g, resolution: 13-bit (DROP)
    #   8-10.	3D-gyroscope data (rad/s)
    #   11-13.	3D-magnetometer data (Î¼T)
    #   14-17.	orientation (invalid in this data collection) (DROP)

    # remove rows with activity label == 0, as per the readme
    data = data[data.iloc[:, 1] != 0]

    # drop heart rate - diff sampling freq
    data = data.drop(columns=[2, (4-1+5-1), (4-1+6-1), (4-1+7-1), (4-1+14-1), (4-1+15-1), (4-1+16-1), (4-1+17-1),
                              (21-1+5-1), (21-1+6-1), (21-1+7-1), (21-1+14-1), (21-1+15-1), (21-1+16-1), (21-1+17-1),
                              (38-1+5-1), (38-1+6-1), (38-1+7-1), (38-1+14-1), (38-1+15-1), (38-1+16-1), (38-1+17-1)])

    print(data.shape)

    # fill nans with 0
    data.fillna(0, inplace=True)

    y = data.iloc[:, 1].to_numpy()
    x = data.iloc[:, 2:].to_numpy()

    # go to 50 Hz
    y = y[::2]
    x = x[::2]
    # fs = 100
    fs = 50

    # window_seconds = 5.12
    window_seconds = 3
    window_length = int(window_seconds * fs)
    shift_length = int(window_length)

    data = []
    labels = []

    for idx in range(window_length, x.shape[0], shift_length):
        first_label = y[idx-window_length]
        last_label = y[idx-1]
        if first_label == last_label:
            window = np.transpose(x[idx - window_length:idx, :])
            data.append(window)
            labels.append(label_map[first_label])

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


