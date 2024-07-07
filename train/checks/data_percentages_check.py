import csv
import os

# Changed stopping percentage check to check for steering, stopping/braking, and running

def calculate_data_percentages(csv_dir):
    total_frames = 0
    stopping_frames = 0
    braking_frames = 0
    steering_frames = 0
    running_frames = 0

    csv_files = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in directory: {csv_dir}")
        return 0, 0

    csv_file = os.path.join(csv_dir, csv_files[0]) 

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            steer_value = float(row[0])
            throttle_value = float(row[1])
            brake_value = float(row[2])
            total_frames += 1
            if steer_value > 0.05:
                steering_frames += 1
            if throttle_value > 0.0 and steer_value < 0.05:
                running_frames += 1
            if brake_value > 0.0 and brake_value < 1.0:
                braking_frames += 1
            if brake_value == 1:
                stopping_frames += 1

    if total_frames == 0:
        print("No data found in the CSV file.")
        return

    return steering_frames, running_frames, braking_frames, stopping_frames, total_frames

csv_directory_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Consolidated Training Data\\csv'
steering_frames, running_frames, braking_frames, stopping_frames, total_frames = calculate_data_percentages(csv_directory_path)
stopping_percentage = (stopping_frames / total_frames) * 100
braking_percentage = (braking_frames / total_frames) * 100
steering_percentage = (steering_frames / total_frames) * 100
running_percentage = (running_frames / total_frames) * 100
print(f"Stopping percentage: {stopping_percentage:.2f}%")
print(f"Total Stopping: {stopping_frames}")
print(f"Braking percentage: {braking_percentage:.2f}%")
print(f"Total Braking: {braking_frames}")
print(f"Steering percentage: {steering_percentage:.2f}%")
print(f"Total Steering: {steering_frames}")
print(f"Running percentage: {running_percentage:.2f}%")
print(f"Total Running: {running_frames}")
print(f"Total Frames: {total_frames}")

# def check_all_directories(root_directory):
#     total_frames = 0
#     stopped_frames = 0
#     steering_frames = 0
#     running_frames = 0
#     for dir_name in os.listdir(root_directory):
#         full_path = os.path.join(root_directory, dir_name, "csv\\")
#         if os.path.isdir(full_path):
#             steer, run, stop, total = calculate_stopping_percentage(full_path)
#             stopped_frames += stop
#             running_frames += run
#             steering_frames += steer
#             total_frames += total
#     stopping_percentage = (stopped_frames / total_frames) * 100
#     print(stopped_frames)
#     print(total_frames)
#     print(f"Stopping percentage: {stopping_percentage:.2f}%")