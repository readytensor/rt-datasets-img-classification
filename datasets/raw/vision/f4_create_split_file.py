"""
This script is used to create the train_test_split.json file for the Vision dataset.
It was used to verify that we get the same splits when the script is re-run.

Also, the contents of the JSON file (actual splits) can be compared with the contents of the 
CSV file (original intended splits). 
"""

import os
import json
from pathlib import Path


from constants import (
    RAW_DIR,
    PROCESSED_VISION_DIR
)


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


def generate_data_split_file():
    file_paths = list_jpg_paths(PROCESSED_VISION_DIR)
    files_info = [file_info(file) for file in file_paths]

    data = {"training": [], "validation": [], "testing": []}

    for file, dir in files_info:
        if dir == "training":
            data["training"].append(file)
        elif dir == "validation":
            data["validation"].append(file)
        elif dir == "testing":
            data["testing"].append(file)

    json_file_path = os.path.join(RAW_DIR, "vision", "train_test_split.json")
    with open(json_file_path, "w") as file:
        json.dump(data, file)

    print("train_test_split.json created successfully")


if __name__ == "__main__":
    generate_data_split_file()
