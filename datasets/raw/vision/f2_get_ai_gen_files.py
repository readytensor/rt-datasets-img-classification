# import os
# from pathlib import Path

# from f3_get_ai_gen_files import get_file_names_and_labels


# RAW_DIR = os.path.join(".", "..", "..", "raw")
# PROCESSED_DIR = os.path.join(".", "..", "..", "processed")

# DATA_DIR = os.path.join(RAW_DIR, "vision", "data")
# AI_GENERATED_DIR_NAMES = ["D36_Photoshop_Generative"]


# def download_ai_gen_files():
#     pass

# def rename_images(X):
#     new_paths = []
#     for i, path in enumerate(X):
#         image_name = Path(path).name
#         new_path = path.replace(image_name, f"{str(i)}.jpg")
#         os.rename(path, new_path)
#         new_paths.append(new_path)
#     return new_paths

# def process_ai_gen_files():

#     # download AI generated files
#     download_ai_gen_files()

#     # get load data
#     X_ai, y_ai = get_file_names_and_labels(
#         ["flat", "nat", "natFBH", "natFBL", "natWA"], include=AI_GENERATED_DIR_NAMES
#     )

#     # rename images
#     X_ai = rename_images(X_ai)
