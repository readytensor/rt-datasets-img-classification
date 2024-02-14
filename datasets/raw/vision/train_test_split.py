import os
import json
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_DIR = "./data"
PROCESSED_VISION_DIR = "../../processed/vision"


def get_file_names_and_labels():
    class_labels = [i for i in os.listdir(DATA_DIR) if i.lower().startswith("d")]
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
        if "/nat/" in img_path:
            for v in variations:
                variation_path = img_path.replace("/nat/", f"/{v}/")
                variation_path = variation_path.replace("_nat_", f"_{v}_")
                X.append(variation_path)
                y.append(label)
    return X, y


def create_split_json_file(X_train, X_test):
    train_files = [os.path.join(i.split("/")[2], i.split("/")[5]) for i in X_train]
    test_files = [os.path.join(i.split("/")[2], i.split("/")[5]) for i in X_test]
    data = {"train": train_files, "test": test_files}
    with open("train_test_split.json", "w") as file:
        json.dump(data, file)

    print("train_test_split.json created successfully!")


def move_file(file_path, split_name, class_label):
    destination_path = f"{PROCESSED_VISION_DIR}/{split_name}/{class_label}"
    os.makedirs(destination_path, exist_ok=True)
    shutil.move(file_path, destination_path)


if __name__ == "__main__":

    X, y = get_file_names_and_labels()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train, y_train = add_image_variations(X_train, y_train)
    X_test, y_test = add_image_variations(X_test, y_test)

    for file_path, label in tqdm(zip(X_train, y_train), desc="Moving train files..."):
        move_file(file_path=file_path, split_name="train", class_label=label)

    for file_path, label in tqdm(zip(X_test, y_test), desc="Moving test files..."):
        move_file(file_path=file_path, split_name="test", class_label=label)

    create_split_json_file(X_train, X_test)
