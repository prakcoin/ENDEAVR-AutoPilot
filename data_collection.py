import argparse
import os
import logging
import numpy as np
import h5py
import carla
import matplotlib.pyplot as plt
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, 
                                road_option_to_int, cleanup, update_spectator, read_routes, 
                                set_traffic_lights_green, get_traffic_light_status, traffic_light_to_int, 
                                CropCustom)
from utils.sensors import start_camera, start_collision_sensor, start_lane_invasion_sensor
from utils.agents import NoisyTrafficManagerAgent, DefaultTrafficManagerAgent

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

def end_episode(ego_vehicle, end_point, frame, idle_frames, args):
    done = False
    if end_reached(ego_vehicle, end_point):
        print("Target reached, episode ending")
        done = True
    elif frame >= args.max_frames:
        print("Maximum frames reached, episode ending")
        done = True
    elif idle_frames >= (args.max_frames / 2):
        print("Vehicle idle for too long, ending episode.")
        done = True
    elif has_collision:
        print("Collision detected, episode ending")
        done = True
    elif has_lane_invasion:
        print("Lane invasion detected, episode ending")
        done = True
    return done

def update_data_file(episode_data, episode_count):
    if not os.path.isdir(f'data'):
        os.makedirs(f'data')

    with h5py.File(f'data/episode_{episode_count + 1}.h5', 'w') as file:
        for key, data_array in episode_data.items():
            data_array = np.array(data_array)
            file.create_dataset(key, data=data_array, maxshape=(None,) + data_array.shape[1:])

def run_episode(world, episode_count, ego_vehicle, agent, vehicle_list, rgb_cam, end_point, args):
    global has_collision
    has_collision = False
    global has_lane_invasion
    has_lane_invasion = False

    episode_data = {
        'rgb': [],
        'controls': [],
        'speed': [],
        'hlc': [],
        'light': [],
    }

    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    frame = 0
    idle_frames = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, idle_frames, args):
            break

        update_spectator(spectator, ego_vehicle)
        
        control, noisy_control = agent.run_step()
        if noisy_control:
            ego_vehicle.apply_control(noisy_control)

        sensor_data = to_rgb(rgb_cam.get_sensor_data())

        velocity = ego_vehicle.get_velocity()
        speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        if speed_km_h == 0.0:
            idle_frames += 1
        else:
            idle_frames = 0

        if not agent.noise:
            frame_data = {
                'rgb': np.array(sensor_data),
                'controls': np.array([control.steer, control.throttle, control.brake]),
                'speed': np.array([speed_km_h]),
                'hlc': np.array([road_option_to_int(agent.get_next_action())]),
                'light': np.array([traffic_light_to_int(get_traffic_light_status(ego_vehicle))])
            }
            if args.collect_steer:
                if abs(control.steer) >= 0.05:
                    for key, value in frame_data.items():
                        episode_data[key].append(value)
            else:
                for key, value in frame_data.items():
                    episode_data[key].append(value)

        world.tick()
        frame += 1

    if not has_collision and not has_lane_invasion and frame <= args.max_frames and idle_frames < (args.max_frames / 2):
        update_data_file(episode_data, episode_count)

def main(args):
    world, client = init_world(args.town)
    traffic_manager = setup_traffic_manager(client)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    world.tick()

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
        if args.noisy_agent:
            agent = NoisyTrafficManagerAgent(ego_vehicle, traffic_manager)
        else:
            agent = DefaultTrafficManagerAgent(ego_vehicle, traffic_manager)
        agent.set_route(route, end_point)

        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager)

        rgb_cam = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors = [rgb_cam.get_sensor(), collision_sensor]
        if args.lane_invasion:
            lane_invasion_sensor = start_lane_invasion_sensor(world, ego_vehicle)
            lane_invasion_sensor.listen(lane_invasion_callback)
            sensors.append(lane_invasion_sensor)
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, episode, ego_vehicle, agent, vehicle_list, rgb_cam, end_point, args)
        if (has_collision or has_lane_invasion):
            num_tries += 1
            episode -= 1
            restart = True
            print("Redoing ", end="")
        else:
            restart = False
        cleanup(client, ego_vehicle, vehicle_list, sensors)
        episode += 1
    print("Simulation complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('--weather', type=str, default='ClearNoon', help='CARLA weather conditions to use')
    parser.add_argument('--max_frames', type=int, default=8000, help='Number of frames to collect per episode')
    parser.add_argument('--episodes', type=int, default=4, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--route_file', type=str, default='routes/Town01_Val.txt', help='Filepath for route file')
    parser.add_argument('--noisy_agent', action="store_true", help='Use noisy agent over default agent')
    parser.add_argument('--lane_invasion', action="store_true", help='Activate lane invasion sensor')
    parser.add_argument('--collect_steer', action="store_true", help='Only collect steering data')
    args = parser.parse_args()

    # logging.basicConfig(filename='data_collection_log.log', 
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s' ) 

    main(args)