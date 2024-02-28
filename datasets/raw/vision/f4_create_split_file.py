import os
import json
from pathlib import Path

RAW_DIR = os.path.join(".", "..", "..", "raw")
PROCESSED_DIR = os.path.join(".", "..", "..", "processed")


def list_jpg_paths(directory):
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                paths.append(os.path.join(root, file))
    return paths


def file_info(file_path):
    path = Path(file_path)
    file_name = path.name
    parent_dir = path.parent
    upper_dir = parent_dir.parent.name
    return file_name, upper_dir


if __name__ == "__main__":
    RAW_DATA_DIR = os.path.join(RAW_DIR, "vision")
    PROCESSED_DATA_DIR = os.path.join(PROCESSED_DIR, "vision")
    file_paths = list_jpg_paths(PROCESSED_DATA_DIR)
    files_info = [file_info(file) for file in file_paths]

    data = {"train": [], "validation": [], "test": []}

    for file, dir in files_info:
        if dir == "training":
            data["train"].append(file)
        elif dir == "validation":
            data["validation"].append(file)
        elif dir == "testing":
            data["test"].append(file)

    json_file_path = os.path.join(RAW_DATA_DIR, "train_test_split.json")
    with open(json_file_path, "w") as file:
        json.dump(data, file)

    print("train_test_split.json created successfully")
