import numpy as np
from moabb.datasets import Zhou2016
from moabb.paradigms import MotorImagery


class_mapping = {"feet": 0,
                 "left_hand": 1,
                 "right_hand": 2}


def load_zhou(subject: str):
    data = Zhou2016()
    paradigm = MotorImagery(resample=100, fmin=8, fmax=30)
    X, labels, _ = paradigm.get_data(data, subjects=[int(subject)])
    labels = np.array([class_mapping[i] for i in labels])
    return X, labels








