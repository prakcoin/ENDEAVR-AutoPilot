import cv2
import os
import re
import pandas as pd

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def play_frames_as_video(img_directory, csv_file):
    image_files = [file for file in os.listdir(img_directory) if file.endswith('.jpg') or file.endswith('.png')]
    image_files = sorted_nicely(image_files)
    
    if not image_files:
        print(f"No image files found in directory: {img_directory}")
        return
    
    df = pd.read_csv(csv_file, header=None, names=['steering_angle', 'throttle', 'brake', 'frame'])
    df.set_index('frame', inplace=True)  # Set the frame column as the index for easy lookup

    playback_speed = 1
    frame_index = 0

    while frame_index < len(image_files):
        image_file = image_files[frame_index]
        frame_number = int(re.search(r'\d+', image_file).group())
        
        if frame_number in df.index:
            steering_angle = df.at[frame_number, 'steering_angle']
            throttle = df.at[frame_number, 'throttle']
            brake = df.at[frame_number, 'brake']

            image_path = os.path.join(img_directory, image_file)
            frame = cv2.imread(image_path)

            text = f"Frame: {frame_number}, Steering Angle: {steering_angle:.2f}, Throttle: {throttle:.2f}, Brake: {brake:.2f}"
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
                frame_index = min(frame_index + 10, len(image_files) - 1)
            elif key & 0xFF == ord('a'):  # Skip back
                frame_index = max(frame_index - 10, 0)
            else:
                frame_index += 1

    cv2.destroyAllWindows()

img_directory_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Training Data\\DownsampleTesting\\Town01_ClearNoon\\img'
csv_file_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Training Data\\DownsampleTesting\\Town01_ClearNoon\\csv\\filtered_data.csv'
play_frames_as_video(img_directory_path, csv_file_path)