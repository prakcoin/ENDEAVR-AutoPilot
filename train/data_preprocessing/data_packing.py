import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import csv
from sklearn.model_selection import train_test_split
from PIL import Image
import h5py

train_image_data_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Consolidated Training Data\\data'
train_csv_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Consolidated Training Data\\csv\\consolidated_data.csv'

val_image_data_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Consolidated Validation Data\\data'
val_csv_path = 'C:\\Users\\User\\Documents\\AV Research\\Data\\Consolidated Validation Data\\csv\\consolidated_data.csv'

# Display single image (debug)
img_example = Image.open(os.path.join(train_image_data_path, os.listdir(train_image_data_path)[2]))
img_example = img_example.convert('RGB')
width, height = img_example.size
top = int(height / 2.05)
height = int(height / 1.05)
cropped_example = img_example.crop((0, top, width, height))
print(np.array(cropped_example))
plt.imshow(cropped_example)
plt.show()

def process_data(image_data_path, csv_path):
    data_dict = {}

    with open(csv_path) as file:
        csv_file = csv.reader(file, delimiter=',')
        next(csv_file)
        for row in csv_file:
            frame_number = row[3]
            img = Image.open(os.path.join(image_data_path, f"{frame_number}.png"))
            img = img.convert('RGB')
            width, height = img.size
            top = int(height / 2.05)
            height = int(height / 1.05)
            cropped_img = img.crop((0, top, width, height))
            cropped_img = np.array(cropped_img)
            steering_angle, throttle, brake = float(row[0]), float(row[1]), float(row[2])
            data_dict[frame_number] = {'image': cropped_img, 'label': (steering_angle, throttle, brake)}

    data = []
    labels = []
    for frame_number, data_info in data_dict.items():
        data.append(data_info['image'])
        labels.append(data_info['label'])

    return data, labels

train_data, train_labels = process_data(train_image_data_path, train_csv_path)
val_data, val_labels = process_data(val_image_data_path, val_csv_path)

with h5py.File('C:\\Users\\User\\Documents\\AV Research\\Data\\train_data.h5', 'w') as f:
    f.create_dataset('data', data=np.array(train_data))
    f.create_dataset('labels', data=np.array(train_labels))

with h5py.File('C:\\Users\\User\\Documents\\AV Research\\Data\\val_data.h5', 'w') as f:
    f.create_dataset('data', data=np.array(val_data))
    f.create_dataset('labels', data=np.array(val_labels))