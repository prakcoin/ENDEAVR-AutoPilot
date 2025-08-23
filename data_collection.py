import argparse
import os
import numpy as np
import h5py
import carla
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb,
                                road_option_to_int, cleanup, update_spectator, read_routes, 
                                spawn_pedestrians, cleanup_pedestrians, align, lidar_to_histogram_features)
from utils.sensors import start_camera, start_collision_sensor, start_lidar_sensor
from utils.agents import NoisyImitationLearningAgent
from utils.vlm_utils import lidar_to_ego_coordinate, align_lidar, normalize_angle_degree

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

def end_episode(ego_vehicle, end_point, frame, args):
    done = False
    if end_reached(ego_vehicle, end_point):
        print("Target reached, episode ending")
        done = True
    elif frame >= args.max_frames:
        print("Maximum frames reached, episode ending")
        done = True
    elif has_collision:
        print("Collision detected, episode ending")
        done = True
    return done

def update_data_file(episode_data, episode_count):
    if not os.path.isdir(f'data'):
        os.makedirs(f'data')

    with h5py.File(f'data/episode_{episode_count + 1 + 64}.h5', 'w') as file:
        for key, data_array in episode_data.items():
            data_array = np.array(data_array)
            file.create_dataset(key, data=data_array, maxshape=(None,) + data_array.shape[1:])

def run_episode(world, episode_count, ego_vehicle, agent, rgb_cam, lidar_sensor, end_point, args):
    global has_collision
    has_collision = False

    episode_data = {
        'rgb': [],
        'lidar': [],
        'controls': [],
        'speed': [],
        'hlc': [],
    }

    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    last_lidar = None
    last_ego_transform = None

    frame = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, args):
            break

        update_spectator(spectator, ego_vehicle)
        
        control, noisy_control = agent.run_step()
        if noisy_control:
            ego_vehicle.apply_control(noisy_control)

        rgb_data = to_rgb(rgb_cam.get_sensor_data())
        lidar_data = lidar_to_ego_coordinate(lidar_sensor.get_sensor_data())

        if last_lidar is not None:
            ego_transform = ego_vehicle.get_transform()
            ego_location = ego_transform.location
            last_ego_location = last_ego_transform.location
            relative_translation = np.array([
                    ego_location.x - last_ego_location.x, ego_location.y - last_ego_location.y,
                    ego_location.z - last_ego_location.z
            ])

            ego_yaw = ego_transform.rotation.yaw
            last_ego_yaw = last_ego_transform.rotation.yaw
            relative_rotation = np.deg2rad(normalize_angle_degree(ego_yaw - last_ego_yaw))

            orientation_target = np.deg2rad(ego_yaw)
            rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                                                    [np.sin(orientation_target),
                                                                     np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
            relative_translation = rotation_matrix.T @ relative_translation

            lidar_last = align_lidar(last_lidar, relative_translation, relative_rotation)
            lidar_360 = np.concatenate((lidar_data, lidar_last), axis=0)
        else:
            lidar_360 = lidar_data

        velocity = ego_vehicle.get_velocity()
        speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        lidar = lidar_to_histogram_features(lidar_360, False)

        if not agent.noise:
            frame_data = {
                'rgb': np.array(rgb_data),
                'lidar': np.array(lidar),
                'controls': np.array([control.steer, control.throttle, control.brake]),
                'speed': np.array([speed_km_h]),
                'hlc': np.array([road_option_to_int(agent.get_next_action())]),
            }
            for key, value in frame_data.items():
                episode_data[key].append(value)

        last_ego_transform = ego_vehicle.get_transform()
        last_lidar = lidar_data

        world.tick()
        frame += 1

    if not has_collision and frame <= args.max_frames:
        update_data_file(episode_data, episode_count)

def main(args):
    world, client = init_world(args.town)
    traffic_manager = setup_traffic_manager(client)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    world.tick()

    weather_conditions = [
        "ClearNoon", 
        "MidRainSunset", 
        "CloudyNight", 
        "WetSunset", 
        "HardRainNoon", 
        "SoftRainNight",
    ]

    route_configs = read_routes(args.route_file)
    episode_count = args.episodes

    vehicle_list = []
    restart = False
    episode = 0
    while episode < episode_count:
        print(f'Episode: {episode + 1}')
        if not restart:
            num_tries = 0
            spawn_point_index, end_point_index, _, route = create_route(route_configs)
        
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[spawn_point_index]
        end_point = spawn_points[end_point_index]

        print(f"Route from spawn point #{spawn_point_index} to #{end_point_index}")

        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        agent = NoisyImitationLearningAgent(ego_vehicle, traffic_manager)
        agent.set_route(route, end_point)

        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager, cars_only=False)
        if (args.pedestrians > 0):
            all_id, all_actors, _ = spawn_pedestrians(world, client, args.pedestrians)

        rgb_cam = start_camera(world, ego_vehicle)
        lidar_sensor = start_lidar_sensor(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors = [rgb_cam.get_sensor(), lidar_sensor.get_sensor(), collision_sensor]
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, episode, ego_vehicle, agent, rgb_cam, lidar_sensor, end_point, args)
        if (has_collision):
            num_tries += 1
            episode -= 1
            restart = True
            print("Redoing ", end="")
        else:
            restart = False
        cleanup(client, ego_vehicle, vehicle_list, sensors)
        cleanup_pedestrians(client, all_id, all_actors)
        episode += 1
    print("Simulation complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('--weather', type=str, default='HardRainNoon', help='CARLA weather conditions to use')
    parser.add_argument('--max_frames', type=int, default=8000, help='Number of frames to collect per episode')
    parser.add_argument('--episodes', type=int, default=16, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--pedestrians', type=int, default=40, help='Number of pedestrians present')
    parser.add_argument('--route_file', type=str, default='routes/Town01_Train.txt', help='Filepath for route file')
    args = parser.parse_args()

    main(args)