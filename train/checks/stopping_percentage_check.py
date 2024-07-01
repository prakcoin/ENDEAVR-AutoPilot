import csv
import os

def calculate_stopping_percentage(directory):
    total_frames = 0
    stopped_frames = 0

    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return 0, 0

    csv_file = os.path.join(directory, csv_files[0]) 

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            brake_value = float(row[2])
            total_frames += 1
            if brake_value == 1.0:
                stopped_frames += 1

    if total_frames == 0:
        print("No data found in the CSV file.")
        return

    return stopped_frames, total_frames

def check_all_directories(root_directory):
    stopped_frames = 0
    total_frames = 0
    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name, "csv\\")
        if os.path.isdir(full_path):
            s, t = calculate_stopping_percentage(full_path)
            stopped_frames += s
            total_frames += t
            print(f"Frames in {full_path}: {t}")
    stopping_percentage = (stopped_frames / total_frames) * 100
    print(stopped_frames)
    print(total_frames)
    print(f"Stopping percentage: {stopping_percentage:.2f}%")

all_directory_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Training Data\\'
directory_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Training Data\\DownsampleTesting'
check_all_directories(all_directory_path)