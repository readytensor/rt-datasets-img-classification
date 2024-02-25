import os
import json
import shutil
import paths
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(paths.RAW_DIR, "vision", "data")
PROCESSED_VISION_DIR = os.path.join(paths.PROCESSED_DIR, "vision")
TRAIN_TEST_SPLIT_FILE = os.path.join(paths.RAW_DIR, "vision", "train_test_split.json")

VALIDATION_SIZE = 0.1

# (2/9 = 0.222) We are using this fraction when we are creating the test split after already subtracting the validation set from training set
# This will result in the test set having 20% of the original data
TEST_SIZE = 2 / 9


def is_segment_in_path(segment, path):
    """
    Check if the specified segment is a part of the given path.

    Parameters:
    - segment (str): The directory segment to check.
    - path (str): The full path to check against.

    Returns:
    - bool: True if the segment is part of the path, False otherwise.
    """
    # Create a Path object for the path
    path_obj = Path(path)

    # Iterate through each part of the path to check if the segment exists
    return segment in path_obj.parts


def replace_segment_in_path(original_path, old_segment, new_segment):
    """
    Replace an old segment with a new segment in the given path.

    Parameters:
    - original_path (str): The original path as a string.
    - old_segment (str): The path segment to be replaced.
    - new_segment (str): The new segment to replace the old one.

    Returns:
    - str: The modified path with the segment replaced.
    """
    # Break down the original path into its components
    parts = list(Path(original_path).parts)

    # Replace the old segment with the new segment where found
    modified_parts = [new_segment if part == old_segment else part for part in parts]

    # Reconstruct the path from the modified parts
    # Use Path() to handle different OS path separators automatically
    modified_path = Path(*modified_parts)

    # Convert the Path object back to a string representation
    return str(modified_path)


def get_file_names_and_labels():
    class_labels = [
        i
        for i in os.listdir(DATA_DIR)
        if i.lower().startswith("d") or i.lower().startswith("g")
    ]
    class_labels = sorted(class_labels)

    X = []
    y = []

    for label in class_labels:
        images_dir_path = os.path.join(DATA_DIR, label, "images")

        sub_dirs = ["flat", "nat"]

        sub_dirs_paths = [os.path.join(images_dir_path, i) for i in sub_dirs]
        for path in sub_dirs_paths:
            images_files_names = [
                i
                for i in os.listdir(path)
                if i.lower().endswith(".jpg") or i.lower().endswith(".jpeg")
            ]
            images_files_paths = [os.path.join(path, i) for i in images_files_names]
            images_files_paths = sorted(images_files_paths)

            X.extend(images_files_paths)
            y += [label] * len(images_files_paths)

    return X, y


def add_image_variations(X, y):
    variations = ["natFBH", "natFBL", "natWA"]
    for img_path, label in zip(X, y):
        if is_segment_in_path("nat", img_path):
            for v in variations:
                variation_path = replace_segment_in_path(img_path, "nat", v)
                variation_path = variation_path.replace("_nat_", f"_{v}_")
                X.append(variation_path)
                y.append(label)
    return X, y


def create_split_json_file(X_train, X_valid, X_test):
    train_files = [Path(i).name for i in X_train]
    validation_files = [Path(i).name for i in X_valid]
    test_files = [Path(i).name for i in X_test]
    data = {"train": train_files, "validation": validation_files, "test": test_files}
    with open(TRAIN_TEST_SPLIT_FILE, "w") as file:
        json.dump(data, file)

    print("train_test_split.json created successfully!")


def move_file(file_path, split_name, class_label):
    destination_path = f"{PROCESSED_VISION_DIR}/{split_name}/{class_label}"
    os.makedirs(destination_path, exist_ok=True)
    shutil.move(file_path, destination_path)


if __name__ == "__main__":

    X, y = get_file_names_and_labels()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, stratify=y_train, random_state=42
    )

    X_train, y_train = add_image_variations(X_train, y_train)
    X_valid, y_valid = add_image_variations(X_valid, y_valid)
    X_test, y_test = add_image_variations(X_test, y_test)

    for file_path, label in tqdm(zip(X_train, y_train), desc="Moving train files..."):
        move_file(file_path=file_path, split_name="training", class_label=label)

    for file_path, label in tqdm(
        zip(X_valid, y_valid), desc="Moving validation files..."
    ):
        move_file(file_path=file_path, split_name="validation", class_label=label)

    for file_path, label in tqdm(zip(X_test, y_test), desc="Moving test files..."):
        move_file(file_path=file_path, split_name="testing", class_label=label)

    create_split_json_file(X_train, X_valid, X_test)
