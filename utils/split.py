import os


def get_loso_splits(dataset: str):
    files = os.listdir(f"datasets/processed/{dataset}")
    subjects = list(set(filename.split("_")[0] for filename in files))
    subjects.sort()
    splits_dict = {}
    for i, test_subject in enumerate(subjects):
        train_split = [s for s in subjects if s != test_subject]
        splits_dict[f"split_{i}"] = {"train": train_split, "test": [test_subject]}
    return splits_dict
