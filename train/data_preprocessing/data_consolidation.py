import os
import shutil
import pandas as pd

root_dir = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Training Data\\'
destination_dir = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Consolidated Training Data\\'

os.makedirs(os.path.join(destination_dir, 'data'), exist_ok=True)
os.makedirs(os.path.join(destination_dir, 'csv'), exist_ok=True)

consolidated_csv = pd.DataFrame(columns=['steering_angle', 'throttle', 'brake', 'frame_number'])
new_frame_number_counter = 0

def consolidate_data(subfolder):
    global new_frame_number_counter
    
    data_path = os.path.join(root_dir, subfolder, 'img')
    csv_path = os.path.join(root_dir, subfolder, 'csv')
    
    csv_file = os.path.join(csv_path, f'{subfolder}.csv')
    if not os.path.exists(csv_file):
        print(f"No csv file found in {csv_path}")
        return
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split(',')
            if len(values) != 4:
                print(f"Invalid row in csv {csv_file}: {line.strip()}")
                continue
            
            steering_angle, throttle, brake, frame_number = values
            
            image_file = os.path.join(data_path, f'{frame_number}.png')
            if os.path.exists(image_file):
                new_frame_number = new_frame_number_counter
                new_frame_number_counter += 1
                new_image_file = os.path.join(destination_dir, 'data', f'{new_frame_number}.png')
                
                shutil.copy(image_file, new_image_file)
                
                consolidated_csv.loc[new_frame_number] = [steering_angle, throttle, brake, new_frame_number]
            else:
                print(f"Image file {image_file} not found")

for subfolder in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, subfolder)):
        consolidate_data(subfolder)

consolidated_csv.to_csv(os.path.join(destination_dir, 'csv', 'consolidated_data.csv'), index=False)
print(f"Consolidation complete. All data has been moved to {destination_dir}")
