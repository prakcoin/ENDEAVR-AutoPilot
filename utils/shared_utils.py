import carla
import random
import numpy as np
import torch
from model.AVModel import AVModel
from torchvision.transforms import v2

def init_world(town):
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    client.load_world(town)
    return world, client

def setup_traffic_manager(client):
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    return traffic_manager

def setup_vehicle_for_tm(traffic_manager, ego_vehicle, route):
    ego_vehicle.set_autopilot(True, 8000)
    traffic_manager.set_route(ego_vehicle, route)
    traffic_manager.ignore_lights_percentage(ego_vehicle, 100)
    traffic_manager.ignore_signs_percentage(ego_vehicle, 100)
    traffic_manager.set_desired_speed(ego_vehicle, 30)

def set_red_light_time(world):
    actor_list = world.get_actors()
    for actor_ in actor_list:
        if isinstance(actor_, carla.TrafficLight):
            actor_.set_red_time(1.0)

def create_route(world, episode_configs):
    spawn_points = world.get_map().get_spawn_points()
    episode_config = random.choice(episode_configs)
    episode_configs.remove(episode_config)
    print(f"Route from spawn point #{episode_config[0][0]} to #{episode_config[0][1]}")
    spawn_point = spawn_points[episode_config[0][0]]
    end_point = spawn_points[episode_config[0][1]]
    route_length = episode_config[1]
    route = episode_config[2]
    return spawn_point, end_point, route_length, route

def spawn_ego_vehicle(world, spawn_point):
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'ego')
    ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
    ego_bp.set_attribute('color', ego_color)
    ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
    return ego_vehicle

def update_spectator(spectator, vehicle):
    ego_transform = vehicle.get_transform()
    spectator_transform = carla.Transform(
        ego_transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    )
    spectator.set_transform(spectator_transform)

def road_option_to_int(high_level_command):
    """convert CARLA.RoadOptions to integer"""
    road_option_dict = {
        "LaneFollow": 0,
        "Left": 1,
        "Right": 2,
        "Straight": 3
    }
    return road_option_dict[high_level_command]

def int_to_road_option(high_level_command):
    """convert integer high-level command to Carla.RoadOptions"""
    road_option_dict = {
        0: "LaneFollow",
        1: "Left",
        2: "Right",
        3: "Straight"
    }
    return road_option_dict[high_level_command]

def to_rgb(image):
    image_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_array = np.reshape(image_array, (image.height, image.width, 4))
    image_array = image_array[:, :, :3]
    image_array = image_array[:, :, ::-1]
    image_array = image_array.copy()
    return image_array

def read_routes(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    routes = [((int(line.split()[0]), int(line.split()[1])), int(line.split()[2]), line.split()[3:]) for line in lines]
    return routes

def cleanup(ego_vehicle, rgb_sensor, collision_sensor, lane_invasion_sensor):
    collision_sensor.destroy()
    lane_invasion_sensor.destroy()
    ego_vehicle.destroy()
    rgb_sensor.get_sensor().destroy()

class CropCustom(object):
    def __call__(self, img):
        img = v2.ToPILImage()(img)
        width, height = img.size
        top = int(height / 2.05)
        bottom = int(height / 1.05)
        cropped_img = img.crop((0, top, width, bottom))
        return cropped_img

def load_model(model_path, device):
    model = AVModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model

def model_control(image, model):
    preprocess = v2.Compose([
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=(0.6651, 0.6439, 0.6236,), std=(0.1352, 0.1353, 0.1364,)),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
    
    output = output.detach().cpu().numpy().flatten()
    steer, throttle_brake = output
    throttle_brake = float(throttle_brake)
    throttle, brake = 0.0, 0.0
    if throttle_brake >= 0.5:
        throttle = (throttle_brake - 0.5) / 0.5
    else:
        brake = (0.5 - throttle_brake) / 0.5
    
    steer = (float(steer) * 2.0) - 1.0
    print(f"Steer: {steer} - Throttle: {throttle} - Brake: {brake}")
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)