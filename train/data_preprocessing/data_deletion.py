import os
import csv
import re

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def keep_first_n_images(directory, n):
    image_files = [file for file in os.listdir(os.path.join(directory,'img\\')) if file.endswith('.png')]
    image_files = sorted_nicely(image_files)
    files_to_delete = image_files[n:]

    for file in files_to_delete:
        os.remove(os.path.join(directory, 'img\\', file))

    csv_files = [file for file in os.listdir(os.path.join(directory,'csv\\')) if file.endswith('.csv')]
    csv_file = os.path.join(directory, 'csv\\', csv_files[0])
    
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows[:n])

def check_all_directories(root_directory, n_images):
    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name)
        print(full_path)
        if os.path.isdir(full_path):
            keep_first_n_images(full_path, n_images)

directory_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Validation Data\\'
check_all_directories(directory_path, 1989)