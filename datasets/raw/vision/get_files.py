import os
import requests
import concurrent.futures
import sys
import time

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


if __name__ == "__main__":
    BASE_URL = 'https://lesc.dinfo.unifi.it/VISION/dataset/'
    SAVE_PATH = 'data'
    FOLDER_MAP_FILE = 'folder_map.txt'

    
    start_time = time.time()  # Start timer
    max_files = 50000
    process_folder_map(FOLDER_MAP_FILE, BASE_URL, SAVE_PATH, max_files=50000)

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Time taken for {max_files} files: {elapsed_time / 60:.2f} minutes")
