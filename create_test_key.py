import os
import pandas as pd
import paths
from pathlib import Path


def list_paths(root_dir):
    """lists all dirs/file paths in a given directory."""
    paths = []
    listdir = [i for i in os.listdir(root_dir) if i != ".DS_Store"]
    for entry in listdir:
        full_path = os.path.join(root_dir, entry)
        paths.append(full_path)

    return paths


def get_test_keys_for_dataset(dataset_path):
    """Returns a dictionary of {id: target} for a given dataset by traversing all labels directories."""
    final_test_keys = {}
    test_path = os.path.join(dataset_path, "testing")
    test_labels_dirs = list_paths(test_path)
    for path in test_labels_dirs:
        label = Path(path).name
        file_names = os.listdir(path)
        test_keys = {k: label for k in file_names}
        final_test_keys.update(test_keys)

    return final_test_keys


def create_datasets_test_keys(dataset_paths):
    """Creates test_key.csv files for all datasets."""
    for path in dataset_paths:
        ids = []
        labels = []
        test_keys = get_test_keys_for_dataset(path)

        for id, label in test_keys.items():
            ids.append(id)
            labels.append(label)

        test_keys_df = pd.DataFrame({"id": ids, "target": labels})
        dataset_name = Path(path).name
        file_name = f"{dataset_name}_test_key.csv"
        save_path = os.path.join(path, file_name)
        test_keys_df.to_csv(save_path, index=False)
        print(f"Test keys created for dataset {dataset_name}")


if __name__ == "__main__":
    PROCESSED_DIR_PATH = paths.PROCESSED_DIR
    dataset_names = [i for i in os.listdir(PROCESSED_DIR_PATH) if i != ".DS_Store" and i != "cub_200_2011" and i != "vision"]
    dataset_paths = [os.path.join(PROCESSED_DIR_PATH, i) for i in dataset_names]

    create_datasets_test_keys(dataset_paths)
