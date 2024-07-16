import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle,  start_camera, create_route, to_rgb, 
                                cleanup, update_spectator, read_routes, CropCustom)
from utils.sensors import start_collision_sensor

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

has_collision = False
def collision_callback(data):
    global has_collision
    has_collision = True

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
        print("Target has been reached, episode ending")
        done = True
    elif frame >= max_frames:
        print("Maximum frames reached, episode ending")
        done = True
    elif has_collision:
        print("Collision detected, episode ending")
        done = True
    return done

def run_episode(world, ego_vehicle, rgb_sensor, end_point, max_frames):
    global has_collision
    has_collision = False

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
            'image': sensor_data,
            'controls': np.array([ego_vehicle.get_control().steer, ego_vehicle.get_control().throttle, ego_vehicle.get_control().brake]),
        }
        for key, value in frame_data.items():
            episode_data[key].append(value)

        world.tick()
        frame += 1

    if not has_collision and frame <= max_frames:
        if not os.path.isfile(f'data.h5'):
            with h5py.File(f'data.h5', 'w') as file:
                for key, data_array in episode_data.items():
                    file.create_dataset(key, data=data_array)
        else:
            with h5py.File(f'data.h5', 'r') as file:
                for key, data_array in episode_data.items():
                    file[key].extend(data_array)

        # if not os.path.exists('data'):
        #     os.makedirs('data')
        # with h5py.File(f'data/episode_{episode + 1}.h5', 'w') as file:
        #     for key, data_array in episode_data.items():
        #         file.create_dataset(key, data=data_array)

def main(args):
    world, client = init_world(args.town, args.weather)
    traffic_manager = setup_traffic_manager(client)
    route_configs = read_routes()

    for episode in range(args.episodes):
        spawn_point, end_point, route = create_route(world, route_configs)
        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        rgb_sensor = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        setup_vehicle_for_tm(traffic_manager, ego_vehicle, route)

        print(f'Episode: {episode + 1}')
        run_episode(world, ego_vehicle, rgb_sensor, end_point, args.frames)
        cleanup(ego_vehicle, rgb_sensor, collision_sensor)

    print("Simulation complete")

if __name__ == '__main__':
    towns = ['Town01', 'Town02', 'Town06']
    weather_conditions = ['ClearNoon', 'ClearSunset', 'ClearNight', 'CloudyNoon', 'CloudyNight', 'WetNoon', 
                        'WetSunset', 'WetNight', 'SoftRainNoon', 'SoftRainSunset', 'SoftRainNight', 
                        'MidRainyNoon', 'MidRainSunset', 'MidRainyNight', 'HardRainNoon']

    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('-t', '--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('-w', '--weather', type=str, default='ClearNoon', help='Weather condition to set')
    parser.add_argument('-f', '--frames', type=int, default=5000, help='Number of frames to collect per episode')
    parser.add_argument('-e', '--episodes', type=int, default=5, help='Number of frames to collect per episode')
    args = parser.parse_args()

    main(args)