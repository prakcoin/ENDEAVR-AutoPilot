import argparse
import os
import random
import torch
import numpy as np
import h5py
import carla
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, 
                                road_option_to_int, cleanup, update_spectator, read_routes, 
                                traffic_light_to_int, load_model, model_control, calculate_delta_yaw)
from utils.sensors import start_camera, start_collision_sensor
from utils.agents import DefaultImitationLearningAgent

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen
# Episode format: (Start point, Endpoint), Length, Route

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
    return done

def update_data_file(episode_data, episode_count):
    if not os.path.isdir(f'data'):
        os.makedirs(f'data')

    with h5py.File(f'data/dagger_episode_{episode_count + 1}.h5', 'w') as file:
        for key, data_array in episode_data.items():
            data_array = np.array(data_array)
            file.create_dataset(key, data=data_array, maxshape=(None,) + data_array.shape[1:])

def run_episode(world, episode_count, ego_vehicle, model, agent, route, rgb_cam, depth_cam, end_point, device, args):
    global has_collision
    has_collision = False

    episode_data = {
        'rgb': [],
        'depth': [],
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
        if end_episode(ego_vehicle, end_point, frame, args) or turning_infraction:
            break

        update_spectator(spectator, ego_vehicle)

        rgb_data = np.array(to_rgb(rgb_cam.get_sensor_data()))
        depth_map = np.array(to_rgb(depth_cam.get_sensor_data()))

        velocity = ego_vehicle.get_velocity()
        speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        if speed_km_h < 1.0:
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
        
        if prev_hlc != 0 and hlc == 0:
            print(f'turned {delta_yaw} degrees')
            if 75 < np.abs(delta_yaw) < 180:
                if delta_yaw < 0 and prev_hlc != 1:
                    turning_infraction = True
                elif delta_yaw > 0 and prev_hlc != 2:
                    turning_infraction = True
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
            control = model_control(rgb_data, depth_map, hlc, speed_km_h, light, model, device)
        ego_vehicle.apply_control(control)

        cur_driver_count += 1
        if cur_driver_count >= 20:
            autopilot = not autopilot
            cur_driver_count = 0

        if idle_frames < 50 and autopilot:
            frame_data = {
                'rgb': np.array(rgb_data),
                'depth': np.array(depth_map),
                'controls': np.array([control.steer, control.throttle, control.brake]),
                'speed': np.array([speed_km_h]),
                'hlc': np.array([hlc]),
                'light': np.array([light])
            }
            for key, value in frame_data.items():
                episode_data[key].append(value)

        world.tick()
        frame += 1

    update_data_file(episode_data, episode_count)

def main(args):
    world, client = init_world(args.town)
    traffic_manager = setup_traffic_manager(client)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    world.tick()

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
        agent = DefaultImitationLearningAgent(ego_vehicle, traffic_manager)
        agent.set_route(route, end_point)

        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager)

        rgb_cam, depth_cam = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors = [rgb_cam.get_sensor(), depth_cam.get_sensor(), collision_sensor]
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, episode, ego_vehicle, model, agent, route, rgb_cam, depth_cam, end_point, device, args)
        cleanup(client, ego_vehicle, vehicle_list, sensors)
        episode += 1
    print("Simulation complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection (DAgger) Script')
    parser.add_argument('--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('--weather', type=str, default='ClearNoon', help='CARLA weather conditions to use')
    parser.add_argument('--max_frames', type=int, default=2000, help='Number of frames to collect per episode')
    parser.add_argument('--model', type=str, default='av_model.pt', help='Name of saved model')
    parser.add_argument('--episodes', type=int, default=8, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--route_file', type=str, default='routes/Town01_Train.txt', help='Filepath for route file')
    args = parser.parse_args()

    main(args)