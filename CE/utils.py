import carla
import random
import queue
import os
import csv
import torch
import numpy as np
from sensors import RGBCamera
from model import AVModel
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2

class CropCustom(object):
    def __call__(self, img):
        width, height = img.size
        top = int(height / 2.05)
        bottom = int(height / 1.05)
        cropped_img = img.crop((0, top, width, bottom))
        return cropped_img

def load_model(model_path, device):
    model = AVModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def init_world(town, weather):
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    client.load_world(town)
    world.set_weather(getattr(carla.WeatherParameters, weather))
    return world, client

def setup_traffic_manager(client):
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    return traffic_manager

def set_red_light_time(world):
    actor_list = world.get_actors()
    for actor_ in actor_list:
        if isinstance(actor_, carla.TrafficLight):
            actor_.set_red_time(1.0)
            world.tick()

def create_route(world, num_points=50):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    spawn_point = random.choice(spawn_points)
    spawn_points.remove(spawn_point)
    if len(spawn_points) >= num_points - 1:
        spawn_points = random.sample(spawn_points, num_points)
    route = [point.location for point in spawn_points]
    return spawn_point, route

def spawn_ego_vehicle(world, spawn_point):
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'ego')
    ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
    ego_bp.set_attribute('color', ego_color)
    ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
    return ego_vehicle

preprocess_no_norm = v2.Compose([
    v2.ToPILImage(),
    CropCustom(),
    v2.Resize((119//2, 256//2)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
])

preprocess = v2.Compose([
    v2.ToPILImage(),
    CropCustom(),
    v2.Resize((119//2, 256//2)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=(0.4872, 0.4669, 0.4469,), std=(0.1138, 0.1115, 0.1074,)),
])

def model_control(sensor, model, vehicle):
    image = sensor.get_sensor_data()
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    array = array.copy()
 
    input_tensor = preprocess(array).unsqueeze(0)
    
    #before_tensor = preprocess_no_norm(array)
    #plt.imshow(before_tensor.permute(1, 2, 0))
    #plt.title('Before Norm Image Tensor')
    #plt.axis('off')
    #plt.show()

    #after_tensor = preprocess(array)
    #plt.imshow(after_tensor.permute(1, 2, 0))
    #plt.title('After Norm Image Tensor')
    #plt.axis('off')
    #plt.show()

    with torch.no_grad():
        output = model(input_tensor)
    
    output = output.detach().cpu().numpy().flatten()
    steer, throttle, brake = output
    
    steer = float(steer)
    throttle = float(throttle)
    brake = float(brake)
    if brake < 0.05: brake = 0.0
    # steer = (float(steer) * 2.0) - 1.0
    
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

def start_camera(world, vehicle):
    rgb_cam = RGBCamera(world, vehicle)
    return rgb_cam

def update_spectator(spectator, vehicle):
    ego_transform = vehicle.get_transform()
    spectator_transform = carla.Transform(
        ego_transform.location + carla.Location(z=50),  # Adjust the height as needed
        carla.Rotation(pitch=-90)  # Look straight down
    )
    spectator.set_transform(spectator_transform)
