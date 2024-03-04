import glob


def get_img_file_count(data_dir):
    path = f'{data_dir}/**/*.jpg'
    jpg_files = glob.glob(path, recursive=True)
    image_count = len(jpg_files)
    print(image_count)


if __name__ == "__main__":
    data_dir = "./data"
    data_dir = "./../../processed/vision"
    get_img_file_count(data_dir)