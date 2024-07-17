import h5py

def calculate_data_percentages(h5_file):
    total_frames = 0
    stopping_frames = 0
    braking_frames = 0
    steering_frames = 0
    running_frames = 0

    with h5py.File(h5_file, 'r') as f:
        # Ensure the HDF5 file contains the required dataset
        if 'controls' not in f:
            print("The HDF5 file does not contain 'controls' dataset.")
            return 0, 0

        controls = f['controls']  # Assuming controls is an array of [steering_angle, throttle, brake]

        for row in controls:
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
        print("No data found in the 'controls' dataset.")
        return

    return steering_frames, running_frames, braking_frames, stopping_frames, total_frames

# Path to your HDF5 file
h5_file_path = '/mnt/c/Users/User/Documents/AV Research/data.h5'
steering_frames, running_frames, braking_frames, stopping_frames, total_frames = calculate_data_percentages(h5_file_path)

if total_frames > 0:
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
