import cv2
import h5py

def play_frames_as_video(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # Ensure the HDF5 file contains the required datasets
        if 'image' not in f or 'controls' not in f:
            print("The HDF5 file does not contain 'images' or 'controls' datasets.")
            return
        
        images = f['image']
        controls = f['controls']  # Assuming controls is an array of [steering_angle, throttle, brake]

        # Ensure both datasets have the same number of frames
        num_frames = len(images)
        if num_frames != len(controls):
            print("The number of frames in 'image' and 'controls' datasets do not match.")
            return

        playback_speed = 1
        frame_index = 0

        while frame_index < num_frames:
            frame = images[frame_index]  # Assuming images are stored as NumPy arrays

            # Ensure the image is in the correct format for OpenCV
            if frame.ndim == 3 and frame.shape[2] == 3:  # Check if the image has 3 channels
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

            steering_angle, throttle, brake = controls[frame_index]

            text = f"Frame: {frame_index}, Steering Angle: {steering_angle:.2f}, Throttle: {throttle:.2f}, Brake: {brake:.2f}"
            font_scale = 0.4
            font_thickness = 1
            y0, dy = 20, 15
            for i, line in enumerate(text.split(',')):
                y = y0 + i * dy
                frame = cv2.putText(frame, line.strip(), (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            cv2.imshow('Frame', frame)

            key = cv2.waitKey(30 // playback_speed)

            if key & 0xFF == ord('q'):  # Exit
                break
            elif key & 0xFF == ord('c'):  # Increase speed
                playback_speed = min(playback_speed + 1, 5)
            elif key & 0xFF == ord('x'):  # Decrease speed
                playback_speed = max(playback_speed - 1, 1)
            elif key & 0xFF == ord('s'):  # Skip forward
                frame_index = min(frame_index + 10, num_frames - 1)
            elif key & 0xFF == ord('a'):  # Skip back
                frame_index = max(frame_index - 10, 0)
            else:
                frame_index += 1

        cv2.destroyAllWindows()

# Path to your HDF5 file
h5_file_path = '/mnt/c/Users/User/Documents/AV Research/data.h5'
play_frames_as_video(h5_file_path)
