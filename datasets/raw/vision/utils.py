import os


RAW_DIR = os.path.join(".", "..", "..", "raw")
PROCESSED_DIR = os.path.join(".", "..", "..", "processed")

DATA_DIR = os.path.join(RAW_DIR, "vision", "data")
AI_GENERATED_DIR_NAMES = ["G01_Photoshop_Generative"]


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
