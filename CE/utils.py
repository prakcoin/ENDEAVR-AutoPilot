import carla
import random
import queue
import os
import csv
import torch
import numpy as np
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

def set_red_light_time(world):
    actor_list = world.get_actors()
    for actor_ in actor_list:
        if isinstance(actor_, carla.TrafficLight):
            actor_.set_red_time(1.0)
            world.tick()

def get_spawn_point(world):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    spawn_point = random.choice(spawn_points)
    return spawn_point

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
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

preprocess = v2.Compose([
    v2.ToPILImage(),
    CropCustom(),
    v2.Resize((119//2, 256//2)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.4872, 0.4669, 0.4469,), std=(0.1138, 0.1115, 0.1074,)),
])

def process_image(image, model, vehicle):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4)).copy()
    array = array[:, :, :3]
    array = array[:, :, ::-1]
 
    input_tensor = preprocess(array).unsqueeze(0)
    print(input_tensor.shape)
    
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
    
    print("Output", output[0])
    steer, throttle, brake = output[0].numpy()
    control = carla.VehicleControl()
    control.steer = float(steer)
    control.throttle = float(throttle)
    control.brake = float(brake)
    vehicle.apply_control(control)
    print(vehicle.get_velocity())

def start_camera(world, vehicle, trans, callback, model):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '256')
    camera_bp.set_attribute('image_size_y', '256')
    camera = world.spawn_actor(camera_bp, trans, attach_to=vehicle)
    camera.listen(lambda image: callback(image, model, vehicle))
    return camera

def update_spectator(spectator, vehicle):
    ego_transform = vehicle.get_transform()
    spectator_transform = carla.Transform(
        ego_transform.location + carla.Location(z=50),  # Adjust the height as needed
        carla.Rotation(pitch=-90)  # Look straight down
    )
    spectator.set_transform(spectator_transform)
