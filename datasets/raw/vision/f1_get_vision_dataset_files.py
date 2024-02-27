import os
import shutil
import requests
import concurrent.futures
import time

BASE_URL = 'https://lesc.dinfo.unifi.it/VISION/dataset/'
SAVE_PATH = 'data'
FOLDER_MAP_FILE = 'folder_map.txt'


def is_image_file(line):
    """ Check if the line represents an image file. """
    return line.strip().endswith('.jpg')

def create_directory(path):
    """ Create a directory if it does not exist. """
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, path):
    """ Download a file from a given URL to a specified path. """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=32768):
                    file.write(chunk)
            return f"Downloaded {url}"
        else:
            return f"Error {response.status_code} at {url}"
    except Exception as e:
        return f"Exception downloading {url}: {e}"


def process_folder_map(file_path, base_url, save_path, max_files=500):
    """ Process the folder map and download images using multithreading. """
    current_path = []
    download_tasks = []
    with open(file_path, 'r') as file:
        for line in file:
            if '|--' in line:
                depth = line.index('|--') // 2
                folder_name = line.split('--')[-1].strip()
                current_path = current_path[:depth] + [folder_name]

                if 'videos' in line:  # Skip processing video folders and their contents
                    current_path = current_path[:-1]
        
            if is_image_file(line):
                relative_file_path = os.path.join(*current_path)
                full_url = os.path.join(base_url, relative_file_path)
                full_url = full_url.replace("\\", "/")
                    
                full_path = os.path.join(save_path, relative_file_path)
                create_directory(os.path.dirname(full_path))
                download_tasks.append((full_url, full_path))
                if len(download_tasks) >= max_files:
                    break
                else:
                    current_path = current_path[:depth] + [folder_name]

    # Download files using multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(download_file, url, path): url for url, path in download_tasks}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result.startswith("Error"):
                    print(result)
                else:
                    print(f"{result} downloaded")
            except Exception as e:
                print(f"Error downloading {url}: {e}")


def clear_data_folders(dir_path: str) -> None:
    """
    Creates the data directory if it does not exist and clears its contents if it does.

    Args:
        dir_path (str): The path to directory where the data should be downloaded.

    """
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Remove the directory and its contents, then recreate the directory
        shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Cleared {dir_path} directory.")
    else:
        # If the directory does not exist, create it
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created {dir_path} directory.")


def download_vision_data(max_files=500):
    """ Download the vision data from the specified URL. """
    # Create the data directory if it does not exist
    clear_data_folders(SAVE_PATH)

    # Download the vision data
    start_time = time.time()  # Start timer
    process_folder_map(FOLDER_MAP_FILE, BASE_URL, SAVE_PATH, max_files=max_files)
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time / 60:.2f} minutes")


if __name__ == "__main__":    
    max_files = 500000 # this was used during testing of this script with a small number of files
    download_vision_data(max_files=max_files)