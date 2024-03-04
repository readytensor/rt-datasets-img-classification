import glob


def get_img_file_count(data_dir):
    path = f'{data_dir}/**/*.jpg'
    jpg_files = glob.glob(path, recursive=True)
    image_count = len(jpg_files)
    return image_count


if __name__ == "__main__":
    # raw
    raw_img_count = get_img_file_count(data_dir="./data")
    print(f"Raw image count: {raw_img_count}")
    
    # processed
    processed_img_count = get_img_file_count(data_dir="./../../processed/vision")
    print(f"Processed image count: {processed_img_count}")