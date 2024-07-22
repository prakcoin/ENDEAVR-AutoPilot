import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

def get_hlc_distribution(h5_file_path, dataset_name='hlc'):
    with h5py.File(h5_file_path, 'r') as file:
        hlc_data = file[dataset_name][:]
        unique, counts = np.unique(hlc_data, return_counts=True)
        distribution = dict(zip(unique, counts))
        
    return distribution


def calculate_data_percentages(h5_file):
    total_frames = 0
    steering_frames = 0
    running_frames = 0

    with h5py.File(h5_file, 'r') as f:
        if 'controls' not in f:
            print("The HDF5 file does not contain 'controls' dataset.")
            return 0, 0

        controls = f['controls']

        for row in controls:
            steer_value = float(row[0])
            total_frames += 1
            if steer_value >= 0.05 or steer_value <= -0.05:
                steering_frames += 1
            if steer_value < 0.05 and steer_value > -0.05:
                running_frames += 1

    if total_frames == 0:
        print("No data found in the 'controls' dataset.")
        return

    return steering_frames, running_frames, total_frames

h5_file_path = '/mnt/c/Users/User/Documents/AV Research/balanced_val_data.h5'
steering_frames, running_frames, total_frames = calculate_data_percentages(h5_file_path)

if total_frames > 0:
    steering_percentage = (steering_frames / total_frames) * 100
    running_percentage = (running_frames / total_frames) * 100

    print(f"Steering percentage: {steering_percentage:.2f}%")
    print(f"Total Steering: {steering_frames}")
    print(f"Running percentage: {running_percentage:.2f}%")
    print(f"Total Running: {running_frames}")
    print(f"Total Frames: {total_frames}")

distribution = get_hlc_distribution(h5_file_path, 'hlc')
print("HLC Distribution:", distribution)