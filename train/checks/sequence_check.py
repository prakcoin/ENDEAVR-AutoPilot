import os
import re

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def check_sequence(directory):
    print(directory)
    image_files = [file for file in os.listdir(directory) if file.endswith('.png')]  # Assuming images have .jpg extension
    image_files_sorted = sorted_nicely(image_files)

    numbers = [int(file.split('.')[0]) for file in image_files_sorted]

    expected_numbers = set(range(min(numbers), max(numbers) + 1))
    actual_numbers = set(numbers)
    missing_numbers = expected_numbers - actual_numbers

    if not missing_numbers:
        print("Files are in sequential order with no missing values.")
    else:
        print("Files are not in sequential order or have missing values.")
        print("Missing values:", sorted(list(missing_numbers)))

def check_all_directories(root_directory):
    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name, "img\\")
        if os.path.isdir(full_path):
            check_sequence(full_path)

directory_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Training Data\\'
check_all_directories(directory_path)