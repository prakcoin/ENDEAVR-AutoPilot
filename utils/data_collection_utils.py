import os
import csv
from .sensors import RGBCamera

def init_dirs_csv(town, weather, episode):
    if not os.path.exists('data'):
        os.makedirs('data')
    run_dir = os.path.join('data', f'{town}_{weather}_{episode}')
    os.makedirs(os.path.join(run_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'csv'), exist_ok=True)

    csv_file = open(os.path.join(run_dir, 'csv', f'{town}_{weather}_{episode}.csv'), 'w+', newline='')
    writer = csv.writer(csv_file)
    return run_dir, writer, csv_file
    
def start_camera(world, vehicle):
    rgb_cam = RGBCamera(world, vehicle, size_x='256', size_y='256')
    return rgb_cam