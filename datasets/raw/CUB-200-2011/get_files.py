import os
import tarfile
import requests
import paths
from tqdm import tqdm

url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
dataset_filename = "CUB_200_2011.tgz"
download_location = os.path.join(paths.RAW_DIR, "CUB-200-2011", "data")
dataset_path = os.path.join(download_location, dataset_filename)
extract_to_path = download_location

os.makedirs(download_location, exist_ok=True)


# Function to download the dataset
def download_dataset(url, dataset_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(dataset_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        print(f"Dataset downloaded to {dataset_path}")


# Download the dataset
download_dataset(url, dataset_path)


# Extract the dataset
def extract_dataset(dataset_path, extract_to_path):
    try:
        with tarfile.open(dataset_path, "r:gz") as tar:
            print("Extracting the dataset. This may take a few minutes...")
            tar.extractall(path=extract_to_path)
            print(f"Dataset extracted to {extract_to_path}")
    except tarfile.ReadError:
        print(
            "The file is not a valid tar.gz archive. Please check the file and try again."
        )


# Extract the dataset
extract_dataset(dataset_path, extract_to_path)
