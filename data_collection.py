import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, create_route, to_rgb, 
                                cleanup, update_spectator, read_routes, CropCustom)
from utils.sensors import start_camera, start_collision_sensor, start_lane_invasion_sensor

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen
# Episode format: (Start point, Endpoint), Length, Route

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

def update_data_file(episode_data):
    if not os.path.isfile(f'data.h5'):
            with h5py.File(f'data.h5', 'w') as file:
                for key, data_array in episode_data.items():
                    data_array = np.array(data_array)
                    file.create_dataset(key, data=data_array, maxshape=(None,) + data_array.shape[1:])
    else:
        with h5py.File(f'data.h5', 'a') as file:
            for key, data_array in episode_data.items():
                data_array = np.array(data_array)
                data_length = len(data_array)
                old_size = file[key].shape[0]
                new_size = old_size + data_length
                file[key].resize(new_size, axis=0)
                file[key][old_size:new_size] = data_array

def run_episode(world, ego_vehicle, rgb_sensor, end_point, max_frames):
    global has_collision
    has_collision = False
    global has_lane_invasion
    has_lane_invasion = False

    episode_data = {
        'image': [],
        'controls': [],
    }

    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    frame = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, max_frames):
            break

        update_spectator(spectator, ego_vehicle)
        sensor_data = to_rgb(rgb_sensor.get_sensor_data())
        sensor_data = CropCustom()(sensor_data)

        frame_data = {
            'image': np.array(sensor_data),
            'controls': np.array([ego_vehicle.get_control().steer, ego_vehicle.get_control().throttle, ego_vehicle.get_control().brake]),
        }
        for key, value in frame_data.items():
            episode_data[key].append(value)

        world.tick()
        frame += 1

    if not has_collision and not has_lane_invasion and frame <= max_frames:
        update_data_file(episode_data)

def main(args):
    world, client = init_world(args.town, args.weather)
    traffic_manager = setup_traffic_manager(client)
    route_configs = read_routes()
    episode_count = min(len(route_configs), args.episodes)

    restart = False
    episode = 0
    while episode < episode_count:
        if not restart:
            spawn_point, end_point, route = create_route(world, route_configs)
        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        rgb_sensor = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        lane_invasion_sensor = start_lane_invasion_sensor(world, ego_vehicle)
        lane_invasion_sensor.listen(lane_invasion_callback)
        setup_vehicle_for_tm(traffic_manager, ego_vehicle, route)

        print(f'Episode: {episode + 1}')
        run_episode(world, ego_vehicle, rgb_sensor, end_point, args.max_frames)
        if (has_collision or has_lane_invasion):
            episode -= 1
            restart = True
            print("Restarting ", end="")
        else:
            restart = False
        cleanup(ego_vehicle, rgb_sensor, collision_sensor)
        episode += 1

    print("Simulation complete")

if __name__ == '__main__':
    towns = ['Town01', 'Town02', 'Town06']
    weather_conditions = ['ClearNoon', 'ClearSunset', 'ClearNight', 'CloudyNoon', 'CloudyNight', 'WetNoon', 
                        'WetSunset', 'WetNight', 'SoftRainNoon', 'SoftRainSunset', 'SoftRainNight', 
                        'MidRainyNoon', 'MidRainSunset', 'MidRainyNight', 'HardRainNoon']

    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('-t', '--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('-w', '--weather', type=str, default='ClearNoon', help='Weather condition to set')
    parser.add_argument('-f', '--max_frames', type=int, default=5000, help='Number of frames to collect per episode')
    parser.add_argument('-e', '--episodes', type=int, default=20, help='Number of frames to collect per episode')
    args = parser.parse_args()

    main(args)