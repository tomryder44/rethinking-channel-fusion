import os
import numpy as np

from paths import RAW_DIR


acc_only = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38]


def load_dsads(subject: str):

    dataset_path = os.path.join(RAW_DIR, "dsads")
    subject_x = []
    subject_y = []
    for y, activity in enumerate(sorted(os.listdir(dataset_path))):  # 19 activities each with a folder
        subject_activity_folder = os.path.join(dataset_path, activity, subject)
        subject_activity_files = sorted(os.listdir(subject_activity_folder))  # 60 text files, one for each segment
        for f in subject_activity_files:
            f = os.path.join(subject_activity_folder, f)
            with open(f, "r") as z:
                z = z.read()
                z = z.split("\n")
                rows = []
                for row in z:
                    if row != "":
                        row = row.split(",")
                        row_data = [float(val) for val in row]
                        rows.append(row_data)
                x = np.transpose(np.array(rows))
                x = x[acc_only, :]
                subject_x.append(x)
                subject_y.append(y)

    data = np.array(subject_x)
    labels = np.array(subject_y)
    return data, labels
