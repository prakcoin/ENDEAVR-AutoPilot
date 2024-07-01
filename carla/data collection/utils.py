import carla
import random
import os
import csv

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

def init_dirs_csv(town, weather):
    # Create directories
    run_dir = os.path.join(f'{town}_{weather}')
    os.makedirs(os.path.join(run_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'csv'), exist_ok=True)

    # Create CSV file for data collection
    csv_file = open(os.path.join(run_dir, 'csv', f'{town}_{weather}.csv'), 'w+', newline='')
    writer = csv.writer(csv_file)
    return run_dir, writer, csv_file

def create_route(world, num_points=50):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    spawn_point = random.choice(spawn_points)
    spawn_points.remove(spawn_point)
    if len(spawn_points) >= num_points - 1:
        spawn_points = random.sample(spawn_points, num_points)
    route = [point.location for point in spawn_points]
    return spawn_point, route

def set_red_light_time(world):
    actor_list = world.get_actors()
    for actor_ in actor_list:
        if isinstance(actor_, carla.TrafficLight):
            actor_.set_red_time(1.0)
            world.tick()

def spawn_ego_vehicle(world, spawn_point):
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'ego')
    ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
    ego_bp.set_attribute('color', ego_color)
    ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
    return ego_vehicle
    
def queue_callback(image, image_queue, control_queue, ego_vehicle):
    image_queue.put(image)
    control_queue.put((ego_vehicle.get_control().steer, ego_vehicle.get_control().throttle, ego_vehicle.get_control().brake, image.frame))

def start_camera(world, vehicle, trans, callback, image_queue, control_queue):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '256')
    camera_bp.set_attribute('image_size_y', '256')
    camera = world.spawn_actor(camera_bp, trans, attach_to=vehicle)
    camera.listen(lambda image: callback(image, image_queue, control_queue, vehicle))
    return camera
