import carla
import random
import numpy as np

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
    # traffic_manager.set_hybrid_physics_mode(True)
    # traffic_manager.set_hybrid_physics_radius(70.0)
    return traffic_manager

def setup_vehicle_for_tm(traffic_manager, ego_vehicle, route):
    ego_vehicle.set_autopilot(True, 8000)
    traffic_manager.set_path(ego_vehicle, route)
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
    spawn_point = spawn_points[episode_config[0][0]]
    end_point = spawn_points[episode_config[0][1]]
    route = episode_config[2]
    return spawn_point, end_point, route

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

def to_rgb(image):
    image_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_array = np.reshape(image_array, (image.height, image.width, 4))
    image_array = image_array[:, :, :3]
    image_array = image_array[:, :, ::-1]
    image_array = image_array.copy()
    return image_array

def read_routes(filename='routes/Town01_All.txt'):
    with open(filename, 'r') as f:
        lines = f.readlines()
    routes = [((int(line.split()[0]), int(line.split()[1])), int(line.split()[2]), line.split()[3:]) for line in lines]
    return routes

def cleanup(ego_vehicle, camera, csv_file=None):
    if csv_file is not None:
        csv_file.close()
    ego_vehicle.destroy()
    camera.get_sensor().destroy()