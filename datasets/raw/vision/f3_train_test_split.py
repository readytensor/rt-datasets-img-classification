"""
This script is used to generate the train, validation and test splits for the Vision data.

For images in "nat", "natFBH", "natFBL", and "natWA" - we ensured that the original image 
from "nat" and its three variations in "natFBH", "natFBL", "natWA" are always retained 
within the same split. This is done to avoid training data leakage into validation and testing 
splits.

The images in "flat" are split randomly into three sets (i.e. there is no special handling
for them). 
"""

import os
import shutil
import random
from pathlib import Path

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from constants import (
    RAW_DATA_DIR,
    PROCESSED_VISION_DIR,
    VALIDATION_SIZE,
    TEST_SIZE,
    TRAIN_TEST_SPLIT_CSV_FILE
)

random.seed(42)


def get_file_names_and_labels():
    class_labels = [
        i
        for i in os.listdir(RAW_DATA_DIR)
        if i.lower().startswith("d") or i.lower().startswith("g")
    ]
    class_labels = sorted(class_labels)

    X = []
    y = []

    for label in class_labels:
        images_dir_path = os.path.join(RAW_DATA_DIR, label, "images")

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


def move_file(file_path, split_name, class_label):
    destination_path = f"{PROCESSED_VISION_DIR}/{split_name}/{class_label}"
    os.makedirs(destination_path, exist_ok=True)
    shutil.move(file_path, destination_path)


def copy_file(file_path, split_name, class_label):
    destination_path = f"{PROCESSED_VISION_DIR}/{split_name}/{class_label}"
    os.makedirs(destination_path, exist_ok=True)
    shutil.copy(file_path, destination_path)


def clear_data_folders(base_dir: str, split_name: str) -> None:
    """
    Clears the contents of the training and testing directories within the specified base
    directory.

    Args:
        base_dir (str): The path to the base directory containing 'training', 'validation'
                        and 'testing' subdirectories.
        split_name (str): Training, validation, or testing.

    Returns:
        None: This function does not return a value but clears specified directories.
    """
    dir_path = os.path.join(base_dir, split_name)
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Remove the directory and its contents, then recreate the directory
        shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Cleared {split_name} directory.")
    else:
        # If the directory does not exist, create it
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created {split_name} directory.")


def write_split_file_names_to_csv(X, y, type_, new_file=False):
    open_type = "w" if new_file else "a"
    with open(TRAIN_TEST_SPLIT_CSV_FILE, open_type) as file:
        if new_file:
            file.write("file_name,label,type\n")
        for file_name_and_path, label in zip(X, y):
            file_name = filename = os.path.basename(file_name_and_path)
            file.write(f"{file_name},{label},{type_}\n")

def create_train_valid_test_splits():

    print("Reading image file names...")
    X, y = get_file_names_and_labels()

    print("Performing train/valid/test splits of file names...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, stratify=y_train, random_state=42
    )

    X_train, y_train = add_image_variations(X_train, y_train)
    X_valid, y_valid = add_image_variations(X_valid, y_valid)
    X_test, y_test = add_image_variations(X_test, y_test)

    splits = {
        "training": (X_train, y_train),
        "validation": (X_valid, y_valid),
        "testing": (X_test, y_test),
    }

    for split_name, (X, y) in splits.items():
        # Save file names to CSV file
        new_file = True if split_name == "training" else False
        write_split_file_names_to_csv(X, y, split_name, new_file=new_file)

        # clear processed folders
        print(f"Clearing previous processed {split_name} data (if any)...")
        clear_data_folders(PROCESSED_VISION_DIR, split_name)

        for file_path, label in tqdm(zip(X, y), desc=f"Copying {split_name} files..."):
            copy_file(file_path=file_path, split_name=split_name, class_label=label)

    print("Train, valid, and test splits performed successfully.")


if __name__ == "__main__":
    create_train_valid_test_splits()
