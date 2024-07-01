import carla
import random

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