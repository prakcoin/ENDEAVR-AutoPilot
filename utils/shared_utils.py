import carla
import random
import numpy as np
import torch
from model.AVModel import AVModelLSTM
import torch.nn.functional as F
from torchvision.transforms import v2

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

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

def setup_vehicle_for_tm(traffic_manager, ego_vehicle):
    ego_vehicle.set_autopilot(True, 8000)
    traffic_manager.ignore_lights_percentage(ego_vehicle, 100)
    traffic_manager.ignore_signs_percentage(ego_vehicle, 100)
    traffic_manager.set_desired_speed(ego_vehicle, 40)

def set_red_light_time(world):
    actor_list = world.get_actors()
    for actor_ in actor_list:
        if isinstance(actor_, carla.TrafficLight):
            actor_.set_red_time(1.0)

def set_traffic_lights_green(world):
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    
    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.set_green_time(9999)
        traffic_light.freeze(True)

def get_traffic_light_status(vehicle):
    light_status = -1
    if vehicle.is_at_traffic_light():
        traffic_light = vehicle.get_traffic_light()
        light_status = traffic_light.get_state()
    return light_status

def traffic_light_to_int(light_status):
    light_dict = {
        -1: 0,
        carla.libcarla.TrafficLightState.Red: 1,
        carla.libcarla.TrafficLightState.Green: 2,
        carla.libcarla.TrafficLightState.Yellow: 3
    }
    return light_dict[light_status]

def create_route(episode_configs):
    episode_config = random.choice(episode_configs)
    #episode_configs.remove(episode_config)
    spawn_point_index = episode_config[0][0]
    end_point_index = episode_config[0][1]
    route_length = episode_config[1]
    route = episode_config[2]
    return spawn_point_index, end_point_index, route_length, route

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def get_vehicle_spawn_points(world, n_vehicles):
    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    if n_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif n_vehicles > number_of_spawn_points:
        print(f'Requested {n_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points')
        n_vehicles = number_of_spawn_points
    return spawn_points

def spawn_ego_vehicle(world, spawn_point):
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'ego')
    ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
    ego_bp.set_attribute('color', ego_color)
    ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
    return ego_vehicle

def spawn_vehicles(world, client, n_vehicles, traffic_manager):
    blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
    blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car'] # cars only
    blueprints = sorted(blueprints, key=lambda bp: bp.id)
    spawn_points = get_vehicle_spawn_points(world, n_vehicles)

    vehicles_list = []
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= n_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(response.error)
        else:
            vehicles_list.append(response.actor_id)
    return vehicles_list

def update_spectator(spectator, vehicle):
    ego_transform = vehicle.get_transform()
    spectator_transform = carla.Transform(
        ego_transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    )
    spectator.set_transform(spectator_transform)

def road_option_to_int(high_level_command):
    road_option_dict = {
        "LaneFollow": 0,
        "Left": 1,
        "Right": 2,
        "Straight": 3
    }
    return road_option_dict[high_level_command]

def int_to_road_option(high_level_command):
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

def calculate_delta_yaw(prev_yaw, cur_yaw):
    delta_yaw = cur_yaw - prev_yaw
    if delta_yaw > 180:
        delta_yaw -= 360
    elif delta_yaw < -180:
        delta_yaw += 360
    return delta_yaw

def cleanup(client, ego_vehicle, vehicles, rgb_sensor, collision_sensor, lane_invasion_sensor):
    ego_vehicle.destroy()
    client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in vehicles])
    rgb_sensor.get_sensor().destroy()
    if collision_sensor:
        collision_sensor.destroy()
    if lane_invasion_sensor:
        lane_invasion_sensor.destroy()

class CropCustom(object):
    def __call__(self, img):
        img = v2.ToPILImage()(img)
        width, height = img.size
        top = int(height / 2.05)
        bottom = int(height / 1.05)
        cropped_img = img.crop((0, top, width, bottom))
        return cropped_img

def load_model(model_path, device):
    model = AVModelLSTM()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model

def model_control(image, hlc, speed, light, model, device):
    input_tensor = torch.tensor(image).permute(2, 0, 1)
    input_tensor = input_tensor / 255.0
    input_tensor = v2.Normalize(mean=(0.7548, 0.7418, 0.7100,), std=(0.1905, 0.1858, 0.1949,))(input_tensor)
    input_tensor = input_tensor.unsqueeze(0)

    hlc = torch.tensor(hlc, dtype=torch.long)
    hlc = F.one_hot(hlc.to(torch.int64), num_classes=4)
    hlc = hlc.unsqueeze(0)

    speed = torch.tensor(speed, dtype=torch.float32)
    speed = torch.clamp(speed / 40, 0, 1)
    speed = speed.unsqueeze(0)

    light = torch.tensor(light, dtype=torch.long)
    light = F.one_hot(light.to(torch.int64), num_classes=4)
    light = light.unsqueeze(0)

    input_tensor = input_tensor.to(device)
    hlc = hlc.to(device)
    speed = speed.to(device)
    light = light.to(device)

    with torch.no_grad():
        output = model(input_tensor, hlc, speed, light)
    
    output = output.detach().cpu().numpy().flatten()
    throttle_brake, steer = output
    throttle_brake = float(throttle_brake)
    throttle, brake = 0.0, 0.0
    if throttle_brake >= 0.5:
        throttle = (throttle_brake - 0.5) / 0.5
    else:
        brake = (0.5 - throttle_brake) / 0.5
    
    steer = (float(steer) * 2.0) - 1.0
    print(f"Steer: {steer}, throttle: {throttle}, brake: {brake}")
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)