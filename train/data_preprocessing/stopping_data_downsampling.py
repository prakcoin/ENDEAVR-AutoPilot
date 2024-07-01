import os
import pandas as pd

def downsample_stopping_data(directory, downsample_ratio=0.5):
    csv_path = os.path.join(directory, "csv")
    csv_files = [file for file in os.listdir(csv_path) if file.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return 0, 0

    csv_file = os.path.join(csv_path, csv_files[0])
    df = pd.read_csv(csv_file, header=None, names=['steering_angle', 'throttle', 'brake', 'frame'])
    stopping_rows = df[df['brake'] == 1.0]
    num_rows_to_keep = int(len(stopping_rows) * downsample_ratio)
    rows_to_keep = stopping_rows.sample(n=num_rows_to_keep)
    rows_to_delete = stopping_rows.drop(rows_to_keep.index)

    for frame_number in rows_to_delete['frame']:
        image_path = os.path.join(directory, 'img\\', f'{frame_number}.png')
        if os.path.exists(image_path):
            os.remove(image_path)

    df_filtered = df.drop(rows_to_delete.index)
    filtered_csv_path = os.path.join(directory, 'csv\\' 'filtered_data.csv')
    df_filtered.to_csv(filtered_csv_path, index=False, header=False)

    print(f"Filtered data saved to {filtered_csv_path}")
    return len(rows_to_delete)

def process_all_directories(root_directory, downsample_ratio=0.5):
    total_stopped_frames = 0

    for dir_name in os.listdir(root_directory):
        path = os.path.join(root_directory, dir_name)
        if os.path.isdir(path):
            stopped = downsample_stopping_data(path, downsample_ratio)
            total_stopped_frames += stopped

    print(f"Total stopped frames deleted: {total_stopped_frames}")

root_directory_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Validation Data\\'
process_all_directories(root_directory_path)