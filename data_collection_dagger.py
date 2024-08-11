import argparse
import os
import logging
import random
import torch
import numpy as np
import h5py
import carla
import matplotlib.pyplot as plt
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, 
                                road_option_to_int, cleanup, update_spectator, read_routes, 
                                set_traffic_lights_green, get_traffic_light_status, traffic_light_to_int, 
                                load_model, model_control, calculate_delta_yaw, CropCustom)
from utils.sensors import start_camera, start_collision_sensor, start_lane_invasion_sensor
from utils.agents import DefaultTrafficManagerAgent

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
    if idle_frames >= (args.max_frames / 2):
        print("Vehicle idle for too long, ending episode.")
        done = True
    return done

def update_data_file(episode_data, episode_count, vehicle_list, args):
    vehicle_str = "novehicles"
    if vehicle_list:
        vehicle_str = "vehicles"

    if not os.path.isdir(f'data'):
        os.makedirs(f'data')

    with h5py.File(f'data/{args.town}_{args.weather}_{vehicle_str}_dagger_episode_{episode_count + 1}.h5', 'w') as file:
        for key, data_array in episode_data.items():
            data_array = np.array(data_array)
            file.create_dataset(key, data=data_array, maxshape=(None,) + data_array.shape[1:])

def run_episode(world, episode_count, ego_vehicle, model, agent, route, vehicle_list, rgb_sensors, end_point, device, args):
    global has_collision
    has_collision = False
    global has_lane_invasion
    has_lane_invasion = False

    episode_data = {
        'image': [],
        'controls': [],
        'speed': [],
        'hlc': [],
        'light': [],
    }

    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    map = world.get_map()
    prev_hlc = 0
    frame = 0
    idle_frames = 0
    cur_driver_count = 0
    prev_hlc = 0
    prev_yaw = 0
    delta_yaw = 0
    turning_infraction = False
    autopilot = False
    while True:
        if end_episode(ego_vehicle, end_point, frame, idle_frames, args) or turning_infraction:
            break

        update_spectator(spectator, ego_vehicle)
    
        sensor_data = to_rgb(rgb_sensors.get_sensor_data())
        #sensor_data = CropCustom()(sensor_data)

        velocity = ego_vehicle.get_velocity()
        speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        if speed_km_h == 0.0:
            idle_frames += 1
        else:
            idle_frames = 0

        vehicle_location = ego_vehicle.get_transform().location
        vehicle_waypoint = map.get_waypoint(vehicle_location)
        next_waypoint = vehicle_waypoint.next(10.0)[0]

        if vehicle_waypoint.is_junction or next_waypoint.is_junction:
            if prev_hlc == 0:
                prev_yaw = ego_vehicle.get_transform().rotation.yaw
                if len(route) > 0:
                    hlc = road_option_to_int(route.pop(0))
                else:
                    hlc = random.choice([1, 2, 3])
            else:
                hlc = prev_hlc
                cur_yaw = ego_vehicle.get_transform().rotation.yaw
                delta_yaw += calculate_delta_yaw(prev_yaw, cur_yaw)
                prev_yaw = cur_yaw
        else:
            hlc = 0
        
        # detect whether the vehicle made the correct turn
        if prev_hlc != 0 and hlc == 0:
            print(f'turned {delta_yaw} degrees')
            # if command is Left or Right but didn't make turn
            if 75 < np.abs(delta_yaw) < 180:
                if delta_yaw < 0 and prev_hlc != 1:
                    turning_infraction = True
                elif delta_yaw > 0 and prev_hlc != 2:
                    turning_infraction = True
            # if command is Go Straight but turned
            elif prev_hlc != 3:
                turning_infraction = True
            if turning_infraction:
                print('Wrong Turn!!!')
            delta_yaw = 0
        
        prev_hlc = hlc

        light_status = -1
        if ego_vehicle.is_at_traffic_light():
            traffic_light = ego_vehicle.get_traffic_light()
            light_status = traffic_light.get_state()
        light = traffic_light_to_int(light_status)
        
        if autopilot:
            control, _ = agent.run_step()
        else:
            control = model_control(sensor_data, hlc, speed_km_h, light, model, device)
        ego_vehicle.apply_control(control)

        # switch between autopilot and model
        cur_driver_count += 1
        if cur_driver_count >= 20:
            autopilot = not autopilot
            cur_driver_count = 0

        if idle_frames < 50 and autopilot:
            frame_data = {
                'image': np.array(sensor_data),
                'controls': np.array([control.steer, control.throttle, control.brake]),
                'speed': np.array([speed_km_h]),
                'hlc': np.array([hlc]),
                'light': np.array([light])
            }
            for key, value in frame_data.items():
                episode_data[key].append(value)

        world.tick()
        frame += 1

    if not has_collision and not has_lane_invasion and frame <= args.max_frames and idle_frames < 6000:
        update_data_file(episode_data, episode_count, vehicle_list, args)

def main(args):
    world, client = init_world(args.town)
    traffic_manager = setup_traffic_manager(client)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    model_path = os.path.join(parent_directory, 'ENDEAVR-AutoPilot', 'model', 'saved_models', args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    print("Current weather:", args.weather)
    route_configs = read_routes(args.route_file)
    episode_count = args.episodes

    vehicle_list = []
    episode = 0
    while episode < episode_count:
        print(f'Episode: {episode + 1}')
        spawn_point_index, end_point_index, _, route = create_route(route_configs)
        
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[spawn_point_index]
        end_point = spawn_points[end_point_index]

        print(f"Route from spawn point #{spawn_point_index} to #{end_point_index}")

        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        agent = DefaultTrafficManagerAgent(ego_vehicle, traffic_manager)
        agent.set_route(route, end_point)

        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager)

        rgb_sensor = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        if args.lane_invasion:
            lane_invasion_sensor = start_lane_invasion_sensor(world, ego_vehicle)
            lane_invasion_sensor.listen(lane_invasion_callback)
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, episode, ego_vehicle, model, agent, route, vehicle_list, rgb_sensor, end_point, device, args)
        if (has_collision or has_lane_invasion):
            episode -= 1
        cleanup(client, ego_vehicle, vehicle_list, rgb_sensor, collision_sensor, None)
        episode += 1
    print("Simulation complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('--weather', type=str, default='ClearNoon', help='CARLA weather conditions to use')
    parser.add_argument('--max_frames', type=int, default=2000, help='Number of frames to collect per episode')
    parser.add_argument('--model', type=str, default='av_model_2.pt', help='Name of saved model')
    parser.add_argument('--episodes', type=int, default=8, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=50, help='Number of vehicles present')
    parser.add_argument('--route_file', type=str, default='routes/Town01_Train.txt', help='Filepath for route file')
    parser.add_argument('--lane_invasion', action="store_true", help='Activate lane invasion sensor')
    args = parser.parse_args()

    # logging.basicConfig(filename='data_collection_log.log', 
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s' ) 

    main(args)