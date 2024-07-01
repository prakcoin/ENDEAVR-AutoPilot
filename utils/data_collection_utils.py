import os
import csv
from sensors import RGBCamera

def init_dirs_csv(town, weather):
    run_dir = os.path.join(f'{town}_{weather}')
    os.makedirs(os.path.join(run_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'csv'), exist_ok=True)

    csv_file = open(os.path.join(run_dir, 'csv', f'{town}_{weather}.csv'), 'w+', newline='')
    writer = csv.writer(csv_file)
    return run_dir, writer, csv_file
    
def queue_callback(image, image_queue, control_queue, ego_vehicle):
    image_queue.put(image)
    control_queue.put((ego_vehicle.get_control().steer, ego_vehicle.get_control().throttle, ego_vehicle.get_control().brake, image.frame))

def start_camera(world, vehicle, callback):
    rgb_cam = RGBCamera(world, vehicle, callback)
    return rgb_cam
