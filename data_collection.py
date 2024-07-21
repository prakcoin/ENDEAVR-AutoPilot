import argparse
import os
import logging
import numpy as np
import h5py
import carla
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, 
                                road_option_to_int, cleanup, update_spectator, read_routes, 
                                set_traffic_lights_green, CropCustom)
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

def update_data_file(episode_data, town, weather, vehicle_list):
    vehicle_str = "novehicles"
    if vehicle_list:
        vehicle_str = "vehicles"

    if not os.path.isdir('data'):
        os.mkdir('data')

    if not os.path.isfile(f'data/{town}_data_{weather}_{vehicle_str}.h5'):
            with h5py.File(f'data/{town}_data_{weather}_{vehicle_str}.h5', 'w') as file:
                for key, data_array in episode_data.items():
                    data_array = np.array(data_array)
                    file.create_dataset(key, data=data_array, maxshape=(None,) + data_array.shape[1:])
    else:
        with h5py.File(f'data/{town}_data_{weather}_{vehicle_str}.h5', 'a') as file:
            for key, data_array in episode_data.items():
                data_array = np.array(data_array)
                data_length = len(data_array)
                old_size = file[key].shape[0]
                new_size = old_size + data_length
                file[key].resize(new_size, axis=0)
                file[key][old_size:new_size] = data_array

def run_episode(world, town, weather, traffic_manager, ego_vehicle, vehicle_list, rgb_sensor, end_point, max_frames):
    global has_collision
    has_collision = False
    global has_lane_invasion
    has_lane_invasion = False

    episode_data = {
        'image': [],
        'controls': [],
        'speed': [],
        'hlc': [],
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

        velocity = ego_vehicle.get_velocity()
        speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        frame_data = {
            'image': np.array(sensor_data),
            'controls': np.array([ego_vehicle.get_control().steer, ego_vehicle.get_control().throttle, ego_vehicle.get_control().brake]),
            'speed': np.array([speed_km_h]),
            'hlc': np.array([road_option_to_int(traffic_manager.get_next_action(ego_vehicle)[0])])
        }
        for key, value in frame_data.items():
            episode_data[key].append(value)

        world.tick()
        frame += 1

    if not has_collision and not has_lane_invasion and frame <= max_frames:
        update_data_file(episode_data, town, weather, vehicle_list)

def main(args):
    weather_conditions = ['ClearNoon', 'WetNoon', 'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon']

    world, client = init_world(args.town)
    traffic_manager = setup_traffic_manager(client)
    set_traffic_lights_green(world)

    for weather in weather_conditions:
        print("Current weather:", weather)
        world.set_weather(getattr(carla.WeatherParameters, weather))
        route_configs = read_routes('routes/Town01_Safe.txt')
        episode_count = min(len(route_configs), args.episodes)

        vehicle_list = []
        restart = False
        episode = 0
        while episode < episode_count:
            if not restart:
                num_tries = 0
                spawn_point_index, end_point_index, _, route = create_route(world, route_configs)
            
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
            setup_vehicle_for_tm(traffic_manager, ego_vehicle, route)

            print(f'Episode: {episode + 1}')
            run_episode(world, args.town, weather, traffic_manager, ego_vehicle, vehicle_list, rgb_sensor, end_point, args.max_frames)
            if (has_collision or has_lane_invasion) and num_tries < 20:
                num_tries += 1
                episode -= 1
                restart = True
                print("Restarting ", end="")
            else:
                restart = False
                if (num_tries == 20):
                    logging.info(f"Skipped episode: Town: {args.town} - Weather: {weather} - Route: {spawn_point_index} to {end_point_index}")
            cleanup(client, ego_vehicle, vehicle_list, rgb_sensor, collision_sensor, lane_invasion_sensor)
            episode += 1


    print("Simulation complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('-t', '--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('-f', '--max_frames', type=int, default=5000, help='Number of frames to collect per episode')
    parser.add_argument('-e', '--episodes', type=int, default=15, help='Number of episodes to collect data for')
    parser.add_argument('-v', '--vehicles', type=int, default=60, help='Number of vehicles present')
    args = parser.parse_args()

    logging.basicConfig(filename='logfile.log', 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s' ) 

    main(args)