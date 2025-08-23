import carla
import random
import numpy as np
import torch
import io
import base64
import re
from model.AVModel import CNNTransformer
from utils.vlm_utils import align_lidar, normalize_angle
import torch.nn.functional as F
from torchvision.transforms import v2

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

def init_world(town):
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return world, client

def setup_traffic_manager(client):
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)
    return traffic_manager

def setup_vehicle_for_tm(traffic_manager, ego_vehicle):
    ego_vehicle.set_autopilot(True)

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
    episode_configs.remove(episode_config)
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
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.filter('model3')[0]
    blueprint.set_attribute('role_name', 'hero')
    ego_vehicle = world.spawn_actor(blueprint, spawn_point)
    return ego_vehicle

def spawn_vehicles(world, client, n_vehicles, traffic_manager, cars_only=True):
    blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
    if cars_only:
        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car'] # cars only
    excluded_bps = {"vehicle.carlamotors.european_hgv", "vehicle.carlamotors.firetruck", "vehicle.carlamotors.carlacola", "vehicle.mitsubishi.fusorosa"}
    blueprints = [x for x in blueprints if x.id not in excluded_bps]
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

def inject_vehicle_noise(world, vehicles_list, traffic_manager):
    for vehicle_id in vehicles_list:
        vehicle = world.get_actor(vehicle_id)
        traffic_manager.ignore_lights_percentage(vehicle, 50)
        traffic_manager.ignore_signs_percentage(vehicle, 50)

def get_pedestrian_spawn_points(world, n):
    spawn_points = []
    for i in range(n):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    return spawn_points

def spawn_pedestrians(world, client, n_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=1.0):
    walkers_list = []
    all_id = []

    spawn_points = get_pedestrian_spawn_points(world, n_pedestrians)
    blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*', '2')

    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            print(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            print(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    world.tick()

    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        all_actors[i].start()
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
    
    return all_id, all_actors, walkers_list

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

def cleanup(client, ego_vehicle, vehicles, sensors):
    ego_vehicle.destroy()
    client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in vehicles])
    for sensor in sensors: sensor.destroy()

def cleanup_pedestrians(client, all_id, all_actors):
    for i in range(0, len(all_id), 2):
        all_actors[i].stop()
    client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

def align(lidar_0, measurements_0, measurements_1, y_augmentation=0.0, yaw_augmentation=0):
    """
    Converts the LiDAR from the coordinate system of measurements_0 to the
    coordinate system of measurements_1. In case of data augmentation, the
    shift of y and rotation around the yaw are taken into account, such that the
    LiDAR is in the same coordinate system as the rotated camera.
    :param lidar_0: (N,3) numpy, LiDAR point cloud
    :param measurements_0: measurements describing the coordinate system of the LiDAR
    :param measurements_1: measurements describing the target coordinate system
    :param y_augmentation: Data augmentation shift in meters
    :param yaw_augmentation: Data augmentation rotation in degree
    :return: (N,3) numpy, Converted LiDAR
    """
    pos_1 = np.array([measurements_1['pos_global'][0], measurements_1['pos_global'][1], 0.0])
    pos_0 = np.array([measurements_0['pos_global'][0], measurements_0['pos_global'][1], 0.0])
    pos_diff = pos_1 - pos_0
    rot_diff = normalize_angle(measurements_1['theta'] - measurements_0['theta'])

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(measurements_1['theta']), -np.sin(measurements_1['theta']), 0.0],
                                [np.sin(measurements_1['theta']),
                                 np.cos(measurements_1['theta']), 0.0], [0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    lidar_1 = align_lidar(lidar_0, pos_diff, rot_diff)

    pos_diff_aug = np.array([0.0, y_augmentation, 0.0])
    rot_diff_aug = np.deg2rad(yaw_augmentation)

    lidar_1_aug = align_lidar(lidar_1, pos_diff_aug, rot_diff_aug)

    return lidar_1_aug

def lidar_to_histogram_features(lidar, use_ground_plane):
    """
    Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
    :param lidar: (N,3) numpy, LiDAR point cloud
    :param use_ground_plane, whether to use the ground plane
    :return: (2, H, W) numpy, LiDAR as sparse image
    """

    def splat_points(point_cloud):
      # 256 x 256 grid
      xbins = np.linspace(-32, 32,
                          (32 - -32) * int(4.0) + 1)
      ybins = np.linspace(-32, 32,
                          (32 - -32) * int(4.0) + 1)
      hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
      hist[hist > 5] = 5
      overhead_splat = hist / 5
      # The transpose here is an efficient axis swap.
      # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
      # (x height channel, y width channel)
      return overhead_splat.T

    # Remove points above the vehicle
    lidar = lidar[lidar[..., 2] < 100.0]
    below = lidar[lidar[..., 2] <= 0.2]
    above = lidar[lidar[..., 2] > 0.2]
    below_features = splat_points(below)
    above_features = splat_points(above)
    if use_ground_plane:
      features = np.stack([below_features, above_features], axis=-1)
    else:
      features = np.stack([above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def load_model(model_path, device):
    model = CNNTransformer()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model

def model_control(rgb, hlc, speed, model, device):
    rgb = torch.tensor(rgb).permute(2, 0, 1)
    rgb = rgb / 255.0
    rgb = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(rgb)
    rgb = rgb.unsqueeze(0)

    hlc = torch.tensor(hlc, dtype=torch.long)
    hlc = F.one_hot(hlc.to(torch.int64), num_classes=4)
    hlc = hlc.unsqueeze(0)

    speed = torch.FloatTensor(speed)
    speed = speed.unsqueeze(0)

    rgb = rgb.to(device)
    hlc = hlc.to(device)
    speed = speed.to(device)

    throttle, steer, brake = inference(model, rgb, hlc, speed)
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

def inference(model, rgb, hlc, speed):
    with torch.no_grad():
        output = model(rgb, hlc, speed)
    
    output = output.detach().cpu().numpy().flatten()
    throttle_brake, steer = output

    throttle_brake = float(throttle_brake)
    throttle, brake = 0.0, 0.0
    if throttle_brake >= 0.5:
        throttle = (throttle_brake - 0.5) / 0.5
    else:
        brake = (0.5 - throttle_brake) / 0.5
    steer = (float(steer) * 2.0) - 1.0

    return throttle, steer, brake

def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def generate_prompt(hlc, speed, steer, brake, throttle):
    road_option = int_to_road_option(hlc)
    road_option_dict = {
        "LaneFollow": "Follow the lane",
        "Left": "Turn left at the junction",
        "Right": "Turn right at the junction",
        "Straight": "Go straight at the junction"
    }    
    lang_hlc = road_option_dict[road_option]

    prompt = (
        "Analyze the following sensor data along with additional context data.\n\n"
        f"- Current high-level command: {lang_hlc}\n"
        f"- Current speed: {speed:.3f} km/h\n"
        f"- Predicted steer value: {steer:.3f}\n"
        f"- Predicted brake value: {brake:.3f}\n"
        f"- Predicted throttle value: {throttle:.3f}\n\n"
        "Determine if the predicted control signals are correct. If correct, confirm them. If incorrect, provide the appropriate values for safe vehicle control."
    )
    return prompt

def parse_chat_response(chat_response):
    chat_content = chat_response.content
    pattern = r"Steer: ([+-]?\d*\.\d+|\d+).*?Brake: ([+-]?\d*\.\d+|\d+).*?Throttle: ([+-]?\d*\.\d+|\d+)"
    match = re.search(pattern, chat_content, re.DOTALL)

    if match:
        steer = float(match.group(1))
        brake = float(match.group(2))
        throttle = float(match.group(3))
    else:
        steer, brake, throttle = 0.0, 1.0, 0.0

    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

def reduce_image_size(image, scale=0.25):
    """Reduce image size by a given scale."""
    original_width, original_height = image.size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    return image.resize((new_width, new_height))

def vlm_inference(openai_client, image, hlc, speed, steer, brake, throttle):
    system_prompt = "You are a powerful vehicle control assistant with the primary responsibility of correcting or confirming vehicle control signals. You will analyze and validate control signals predicted by a convolutional neural network in the CARLA Simulator. You will use the following inputs:\n- Sensor data from a front RGB camera.\n- The current high-level command (one of: 'Follow the lane', 'Turn left at the junction', 'Turn right at the junction', or 'Go straight at the junction').\n- The ego vehicle's current speed in km/h.\n- Steer value (range: -1.0 to 1.0, where positive values indicate a right turn and negative values indicate a left turn).\n- Brake value (range: 0.0 to 1.0, where 0.0 is no braking and 1.0 is full braking, bringing the vehicle to a stop).\n- Throttle value (range: 0.0 to 1.0, where 0.0 is no acceleration and 1.0 is full acceleration).\nWhen validating or correcting control signals, consider the following factors:\n- Environmental Conditions: Weather, lighting, road type, lane markings, etc.\n- Traffic Context: Presence of nearby vehicles, pedestrians, traffic lights, or junctions.\n- High-Level Command: Ensure the control signals align with the intended maneuver (e.g., lane following, turning at a junction, going straight at a junction).\n- Current Speed: Adjust throttle and brake values to maintain safe speeds.\nProvide your response in a structured format, clearly stating whether the predicted signals are correct or incorrect. If incorrect, include the appropriate control signals for safe vehicle operation."
    image = reduce_image_size(image)
    encoded_image = encode_image(image)
    prompt = generate_prompt(hlc, speed, steer, brake, throttle)
    chat_response = openai_client.chat.completions.create(
        model="prakcoin/QwENDEAVR2.5-VL-Base",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
                {"type": "text", "text": prompt}
            ]}
        ],
        temperature=1.0
    )
    response = chat_response.choices[0].message.content
    vlm_control = parse_chat_response(chat_response.choices[0].message)
    return vlm_control, prompt, response