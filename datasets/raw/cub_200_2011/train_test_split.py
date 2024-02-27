import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(paths.RAW_DIR, "cub_200_2011", "data", "cub_200_2011", "images")
PROCESSED_DIR = os.path.join(paths.PROCESSED_DIR, "cub_200_2011")
TRAIN_TEST_SPLIT_FILE = os.path.join(
    paths.RAW_DIR, "cub_200_2011", "train_test_split.json"
)

TEST_SIZE = 0.2


def list_directories(root_dir):
    directories = []
    for root, dirs, _ in os.walk(root_dir):
        for dir_name in dirs:
            directories.append(os.path.join(root, dir_name))
    return directories


def get_file_names_and_labels():
    class_labels = [Path(i).name for i in list_directories(DATA_DIR)]
    class_labels = sorted(class_labels)

    X = []
    y = []

    for label in class_labels:
        images_dir_path = os.path.join(DATA_DIR, label)

        images_files_names = [
            i
            for i in os.listdir(images_dir_path)
            if i.lower().endswith(".jpg") or i.lower().endswith(".jpeg")
        ]
        images_files_paths = [
            os.path.join(images_dir_path, i) for i in images_files_names
        ]
        images_files_paths = sorted(images_files_paths)

        X.extend(images_files_paths)
        y += [label] * len(images_files_paths)

    return X, y


def create_split_json_file(X_train, X_test):
    train_files = [Path(i).name for i in X_train]
    test_files = [Path(i).name for i in X_test]
    data = {"train": train_files, "test": test_files}
    with open(TRAIN_TEST_SPLIT_FILE, "w") as file:
        json.dump(data, file)

    print("train_test_split.json created successfully!")


def copy_file(file_path, split_name, class_label):
    destination_path = f"{PROCESSED_DIR}/{split_name}/{class_label}"
    os.makedirs(destination_path, exist_ok=True)
    shutil.copy(file_path, destination_path)


if __name__ == "__main__":

    X, y = get_file_names_and_labels()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )

    for file_path, label in tqdm(zip(X_train, y_train), desc="Copying train files..."):
        copy_file(file_path=file_path, split_name="training", class_label=label)

    for file_path, label in tqdm(zip(X_test, y_test), desc="Copying test files..."):
        copy_file(file_path=file_path, split_name="testing", class_label=label)

    create_split_json_file(X_train, X_test)
