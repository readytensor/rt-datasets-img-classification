import os


# root directory where raw and processed data is stored for all datasets
RAW_DIR = os.path.join(".", "..", "..", "raw")
PROCESSED_DIR = os.path.join(".", "..", "..", "processed")


# URL from where the data is downloaded
BASE_URL = 'https://lesc.dinfo.unifi.it/VISION/dataset/'

# Path where the folder map file is saved. We use this file to know
# the directory structure of the dataset
FOLDER_MAP_FILE = os.path.join(RAW_DIR, "vision", "folder_map.txt")

# directory where raw Vision data is downloaded from the BASE _URL above
RAW_DATA_DIR = os.path.join(RAW_DIR, "vision", "data")

# directories where processed data is stored
AI_GENERATED_DIR_NAMES = ["G01_Photoshop_Generative"]

# !!!TODO!!!
# Define paths from where the AI generated data is downloaded
# !!!TODO!!!

# directory where processed Vision data is stored
PROCESSED_VISION_DIR = os.path.join(PROCESSED_DIR, "vision")

# splits for train/valid/test = 70/10/20
VALIDATION_SIZE = 0.1
# (2/9 = 0.222) We are using this fraction when we are creating the test split after already
# subtracting the validation set from it.
# This will result in the test set having 20% of the original data
# effectively, we get 70%/10%/20% split for train/valid/test.
TEST_SIZE = 2 / 9

# file where the list of image files that go into each split is saved
TRAIN_TEST_SPLIT_FILE = os.path.join(RAW_DIR, "vision", "train_test_split.json")
TRAIN_TEST_SPLIT_CSV_FILE = os.path.join(RAW_DIR, "vision", "train_test_split.csv")
