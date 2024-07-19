import os
import argparse
import torch
import carla
from utils.sensors import start_camera, start_collision_sensor, start_lane_invasion_sensor
from utils.shared_utils import (init_world, read_routes, create_route,
                                spawn_ego_vehicle, setup_traffic_manager, 
                                cleanup, update_spectator, to_rgb, CropCustom,
                                model_control, load_model)
from utils.dist_tracker import DistanceTracker
from utils.hlc_loader import HighLevelCommandLoader

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

has_collision = False
def collision_callback(data):
    global has_collision
    has_collision = True

has_lane_invasion = False
def lane_invasion_callback(data):
    global has_lane_invasion
    has_lane_invasion = True

def end_reached(ego_vehicle, end_point):
    vehicle_location = ego_vehicle.get_location()
    end_location = end_point.location

    if end_location is None:
        return False 

    distance = vehicle_location.distance(end_location)
    return distance < 1.0

def end_episode(ego_vehicle, end_point, frame, max_frames):
    done = False
    if end_reached(ego_vehicle, end_point):
        print("Target reached, episode ending")
        done = True
    elif frame >= max_frames:
        print("Maximum frames reached, episode ending")
        done = True
    elif has_collision:
        print("Collision detected, episode ending")
        done = True
    elif has_lane_invasion:
        print("Lane invasion detected, episode ending")
        done = True
    return done

def run_episode(world, model, device, ego_vehicle, rgb_sensor, end_point, route, route_length, max_frames):
    global has_collision
    has_collision = False
    global has_lane_invasion
    has_lane_invasion = False

    dist_tracker = DistanceTracker()
    hlc_loader = HighLevelCommandLoader(ego_vehicle, world.get_map(), route)
    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    frame = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, max_frames):
            break
        
        hlc = hlc_loader.get_next_hlc()

        update_spectator(spectator, ego_vehicle)
        sensor_data = to_rgb(rgb_sensor.get_sensor_data())
        sensor_data = CropCustom()(sensor_data)

        control = model_control(sensor_data, hlc, model, device)
        ego_vehicle.apply_control(control)
        dist_tracker.update(ego_vehicle)
        world.tick()
        frame += 1

    if not has_collision and not has_lane_invasion and frame <= max_frames:
        print("Episode successfully completed")
        print("Route completion: 1.0")
        return True

    route_completion = dist_tracker.get_total_distance() / route_length
    print(f"Route completion: {route_completion}")
    return False



def main(args):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    model_path = os.path.join(parent_directory, 'ENDEAVR-AutoPilot', 'model', 'saved_models', 'av_modelv2.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    world, client = init_world(args.town)
    traffic_manager = setup_traffic_manager(client)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    route_configs = read_routes('routes/Town02_Test.txt')
    episode_count = min(len(route_configs), args.episodes)
    completed_episodes = 0
    for episode in range(episode_count):
        spawn_point, end_point, route_length, route = create_route(world, route_configs)

        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        rgb_sensor = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        lane_invasion_sensor = start_lane_invasion_sensor(world, ego_vehicle)
        lane_invasion_sensor.listen(lane_invasion_callback)

        if run_episode(world, model, device, ego_vehicle, rgb_sensor, end_point, route, route_length, args.max_frames):
            completed_episodes += 1

        cleanup(ego_vehicle, rgb_sensor, collision_sensor, lane_invasion_sensor)
    print(f"Episode completion rate: {completed_episodes / episode_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('-t', '--town', type=str, default='Town02', help='CARLA town to use')
    parser.add_argument('-w', '--weather', type=str, default='ClearNoon', help='Weather condition to set')
    parser.add_argument('-f', '--max_frames', type=int, default=5000, help='Number of frames to collect per episode')
    parser.add_argument('-e', '--episodes', type=int, default=4, help='Number of episodes to evaluate for')
    args = parser.parse_args()
    main(args)