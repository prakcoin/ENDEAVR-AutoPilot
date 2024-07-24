import os
import argparse
import torch
import carla
import logging
import numpy as np
from utils.sensors import start_camera, start_collision_sensor, start_lane_invasion_sensor
from utils.shared_utils import (init_world, read_routes, create_route,
                                spawn_ego_vehicle, spawn_vehicles, setup_traffic_manager, 
                                cleanup, update_spectator, to_rgb, CropCustom,
                                model_control, load_model)
from utils.dist_tracker import DistanceTracker
from utils.hlc_loader import HighLevelCommandLoader

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

num_collisions = 0
has_collision = False
def collision_callback(data):
    global num_collisions
    global has_collision
    has_collision = True
    num_collisions += 1

num_lane_invasions = 0
def lane_invasion_callback(data):
    global num_lane_invasions
    num_lane_invasions += 1

def end_reached(ego_vehicle, end_point):
    vehicle_location = ego_vehicle.get_location()
    end_location = end_point.location

    if end_location is None:
        return False 

    distance = vehicle_location.distance(end_location)
    return distance < 1.0

def end_episode(ego_vehicle, end_point, frame, max_frames, idle_frames):
    done = False
    if end_reached(ego_vehicle, end_point):
        print("Target reached, episode ending")
        done = True
    elif frame >= max_frames:
        print("Maximum frames reached, episode ending")
        done = True
    if idle_frames >= 600:
        print("Vehicle idle for too long, ending episode.")
        done = True
    elif has_collision:
        print("Collision detected, episode ending")
        done = True
    return done

def run_episode(world, model, device, ego_vehicle, rgb_sensor, end_point, route, route_length, max_frames):
    global num_collisions
    num_collisions = 0
    global has_collision
    has_collision = False
    global num_lane_invasions
    num_lane_invasions = 0

    dist_tracker = DistanceTracker()
    hlc_loader = HighLevelCommandLoader(ego_vehicle, world.get_map(), route)
    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    frame = 0
    idle_frames = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, max_frames, idle_frames):
            break
        
        velocity = ego_vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_km_h = 3.6 * speed_m_s

        if speed_km_h == 0.0:
            idle_frames += 1
        else:
            idle_frames = 0

        hlc = hlc_loader.get_next_hlc()

        update_spectator(spectator, ego_vehicle)
        sensor_data = to_rgb(rgb_sensor.get_sensor_data())
        sensor_data = np.array(CropCustom()(sensor_data))

        control = model_control(sensor_data, hlc, speed_km_h, model, device)
        ego_vehicle.apply_control(control)
        dist_tracker.update(ego_vehicle)
        world.tick()
        frame += 1

    if not has_collision and frame <= max_frames:
        print("Episode successfully completed")
        print("Route completion: 1.0")
        return True, 1.0

    route_completion = dist_tracker.get_total_distance() / route_length
    print(f"Route completion: {route_completion}")
    return False, route_completion

def main(args):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    model_path = os.path.join(parent_directory, 'ENDEAVR-AutoPilot', 'model', 'saved_models', 'av_modelv3.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    world, client = init_world(args.town)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))

    traffic_manager = setup_traffic_manager(client)
    route_configs = read_routes(args.route_file)
    episode_count = min(len(route_configs), args.episodes)
    
    vehicle_list = []
    completed_episodes = 0
    route_completions = []
    collisions = []
    lane_invasions = []
    for _ in range(episode_count):
        spawn_point_index, end_point_index, route_length, route = create_route(route_configs)
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[spawn_point_index]
        end_point = spawn_points[end_point_index]

        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager)

        rgb_sensor = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        lane_invasion_sensor = start_lane_invasion_sensor(world, ego_vehicle)
        lane_invasion_sensor.listen(lane_invasion_callback)
        
        episode_completed, route_completion = run_episode(world, model, device, ego_vehicle, rgb_sensor, end_point, route, route_length, args.max_frames)
        if episode_completed:
            completed_episodes += 1

        route_completions.append(route_completion)
        collisions.append(num_collisions)
        lane_invasions.append(num_lane_invasions)
        cleanup(client, ego_vehicle, vehicle_list, rgb_sensor, collision_sensor, lane_invasion_sensor)
    logging.info(f"Episode completion rate: {completed_episodes / episode_count}")
    logging.info(f"Average route completion: {sum(route_completions) / len(route_completions)}")
    logging.info(f"Average collisions: {sum(collisions) / len(collisions)}")
    logging.info(f"Average lane invasions: {sum(lane_invasions) / len(lane_invasions)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Model Evaluation Script')
    parser.add_argument('-t', '--town', type=str, default='Town02', help='CARLA town to use')
    parser.add_argument('-w', '--weather', type=str, default='ClearNoon', help='Weather condition to set')
    parser.add_argument('-f', '--max_frames', type=int, default=5000, help='Number of frames to collect per episode')
    parser.add_argument('-e', '--episodes', type=int, default=4, help='Number of episodes to evaluate for')
    parser.add_argument('-v', '--vehicles', type=int, default=0, help='Number of vehicles present')
    parser.add_argument('-r', '--route_file', type=str, default='routes/Town02_Test.txt', help='Filepath for route file')
    args = parser.parse_args()
    
    logging.basicConfig(filename='evaluation_log.log', 
                        level=logging.INFO,
                        format='%(message)s' ) 

    main(args)