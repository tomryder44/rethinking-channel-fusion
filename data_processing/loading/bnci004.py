import numpy as np
from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery


class_mapping = {"left_hand": 0,
                 "right_hand": 1}


def load_bnci004(subject: str):
    data = BNCI2014_004()
    paradigm = MotorImagery(resample=100, fmin=8, fmax=30)
    X, labels, _ = paradigm.get_data(data, subjects=[int(subject)])
    labels = np.array([class_mapping[i] for i in labels])
    return X, labels
