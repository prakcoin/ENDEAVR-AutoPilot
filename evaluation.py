import os
import argparse
import torch
import carla
import logging
import numpy as np
from utils.sensors import start_camera, start_collision_sensor, start_lane_invasion_sensor
from utils.shared_utils import (init_world, read_routes, create_route, traffic_light_to_int,
                                spawn_ego_vehicle, spawn_vehicles, setup_traffic_manager, road_option_to_int, traffic_light_to_int,
                                cleanup, update_spectator, to_rgb, calculate_delta_yaw, CropCustom,
                                model_control, load_model)
from utils.dist_tracker import DistanceTracker
from utils.hlc_loader import HighLevelCommandLoader

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

#Simulation timeout — If no client-server communication can be established in 60 seconds.
#Off-road driving — If an agent drives off-road, that percentage of the route will not be considered towards the computation of the route completion score.

COLLISION_WALKER_PENALTY = 0.5
COLLISION_VEHICLE_PENALTY = 0.6
COLLISION_OTHER_PENALTY = 0.65
RED_LIGHT_PENALTY = 0.7
TIMEOUT_PENALTY = 0.7
WRONG_TURN_PENALTY = 0.7

has_collision = False
collision_type = None
def collision_callback(data):
    global has_collision
    collision_type = type(data.other_actor)
    has_collision = True

num_vehicle_collisions = 0
num_walker_collisions = 0
num_other_collisions = 0
num_red_light_infractions = 0
num_timeouts = 0
num_wrong_turns = 0

def end_reached(ego_vehicle, end_point):
    vehicle_location = ego_vehicle.get_location()
    end_location = end_point.location

    if end_location is None:
        return False 

    distance = vehicle_location.distance(end_location)
    return distance < 1.0

def end_episode(ego_vehicle, end_point, frame, max_frames, idle_frames, turning_infraction):
    global num_timeouts
    done = False
    if end_reached(ego_vehicle, end_point):
        logging.info("Target reached, episode ending")
        done = True
    elif frame >= max_frames:
        logging.info("Maximum frames reached, episode ending")
        num_timeouts += 1
        done = True
    elif idle_frames >= 500:
        logging.info("Vehicle idle for too long, episode ending")
        num_timeouts += 1
        done = True
    elif turning_infraction:
        logging.info("Turning infraction, episode ending")
        done = True
    return done

def check_collision(prev_collision):
    global has_collision, collision_type, num_other_collisions
    if has_collision:
        if not prev_collision:
            if collision_type == carla.libcarla.Vehicle:
                num_vehicle_collisions += 1
            elif collision_type == carla.libcarla.Walker:
                num_walker_collisions += 1
            else:
                num_other_collisions += 1
            prev_collision = True
    else:
        prev_collision = False
    has_collision = False
    collision_type = None
    return prev_collision

def run_episode(world, model, device, ego_vehicle, rgb_cam, end_point, route, route_length, max_frames):
    global has_collision, collision_type
    has_collision = False
    global num_wrong_turns
    num_wrong_turns = 0

    global num_red_light_infractions
    num_red_light_infractions = 0

    dist_tracker = DistanceTracker()
    hlc_loader = HighLevelCommandLoader(ego_vehicle, world.get_map(), route)
    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    frame = 0
    prev_hlc = 0
    prev_yaw = 0
    delta_yaw = 0
    turning_infraction = False
    idle_frames = 0
    running_light = False
    prev_collision = False
    while True:
        prev_collision = check_collision(prev_collision)

        if end_episode(ego_vehicle, end_point, frame, max_frames, idle_frames, turning_infraction):
            break
        
        transform = ego_vehicle.get_transform()
        vehicle_location = transform.location

        velocity = ego_vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_km_h = 3.6 * speed_m_s

        if speed_km_h < 1.0:
            idle_frames += 1
        else:
            idle_frames = 0

        hlc = hlc_loader.get_next_hlc()
        if hlc != 0:
            if prev_hlc == 0:
                prev_yaw = ego_vehicle.get_transform().rotation.yaw
            else:
                cur_yaw = ego_vehicle.get_transform().rotation.yaw
                delta_yaw += calculate_delta_yaw(prev_yaw, cur_yaw)
                prev_yaw = cur_yaw
        
        if prev_hlc != 0 and hlc == 0:
            logging.info(f'turned {delta_yaw} degrees')
            if 60 < np.abs(delta_yaw) < 180:
                if delta_yaw < 0 and prev_hlc != 1:
                    turning_infraction = True
                elif delta_yaw > 0 and prev_hlc != 2:
                    turning_infraction = True
            elif prev_hlc != 3:
                turning_infraction = True
            if turning_infraction:
                num_wrong_turns += 1
            delta_yaw = 0
        
        prev_hlc = hlc

        update_spectator(spectator, ego_vehicle)
        sensor_data = np.array(to_rgb(rgb_cam.get_sensor_data()))

        light_status = -1
        if ego_vehicle.is_at_traffic_light():
            traffic_light = ego_vehicle.get_traffic_light()
            light_status = traffic_light.get_state()
            traffic_light_location = traffic_light.get_transform().location
            distance_to_traffic_light = np.sqrt((vehicle_location.x - traffic_light_location.x)**2 + (vehicle_location.y - traffic_light_location.y)**2)
            if light_status == carla.libcarla.TrafficLightState.Red and distance_to_traffic_light < 6 and speed_m_s > 5:
                if not running_light:
                    running_light = True
                    num_red_light_infractions += 1
            else:
                running_light = False
        light = np.array([traffic_light_to_int(light_status)])

        control = model_control(sensor_data, hlc, speed_km_h, light, model, device)
        ego_vehicle.apply_control(control)
        dist_tracker.update(ego_vehicle)
        world.tick()
        frame += 1

    if end_reached(ego_vehicle, end_point):
        logging.info("Route completion: 1.0")
        return True, 1.0

    route_completion = min(dist_tracker.get_total_distance() / route_length, 1.0)
    logging.info(f"Route completion: {route_completion}")
    return False, route_completion

def main(args):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    model_path = os.path.join(parent_directory, 'ENDEAVR-AutoPilot', 'model', 'saved_models', args.model)
    
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
    infraction_penalties = []
    driving_scores = []

    for episode in range(episode_count):
        spawn_point_index, end_point_index, route_length, route = create_route(route_configs)
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[spawn_point_index]
        end_point = spawn_points[end_point_index]

        logging.info(f"Episode {episode}: route from spawn point #{spawn_point_index} to #{end_point_index}")

        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager)

        rgb_cam = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors = [rgb_cam.get_sensor(), collision_sensor]

        episode_completed, route_completion = run_episode(world, model, device, ego_vehicle, rgb_cam, end_point, route, route_length, args.max_frames)
        if episode_completed:
            completed_episodes += 1

        route_completions.append(route_completion)
        infraction_penalty = COLLISION_VEHICLE_PENALTY ** num_walker_collisions * \
                            COLLISION_WALKER_PENALTY ** num_walker_collisions * \
                            COLLISION_OTHER_PENALTY ** num_other_collisions * \
                            RED_LIGHT_PENALTY ** num_red_light_infractions * \
                            TIMEOUT_PENALTY ** num_timeouts * \
                            WRONG_TURN_PENALTY ** num_wrong_turns
        infraction_penalties.append(infraction_penalty)
        driving_score = infraction_penalty * route_completion
        driving_scores.append(driving_score)
        cleanup(client, ego_vehicle, vehicle_list, sensors)

    logging.info(f"Episode completion rate: {completed_episodes / episode_count}")
    logging.info(f"Average route completion: {sum(route_completions) / episode_count}")
    logging.info(f"Average infraction penalty: {sum(infraction_penalties) / episode_count}")
    logging.info(f"Average driving score: {sum(driving_scores) / episode_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Model Evaluation Script')
    parser.add_argument('--town', type=str, default='Town02', help='CARLA town to use')
    parser.add_argument('--weather', type=str, default='ClearNoon', help='Weather condition to set')
    parser.add_argument('--max_frames', type=int, default=8000, help='Number of frames before terminating episode')
    parser.add_argument('--episodes', type=int, default=12, help='Number of episodes to evaluate for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--route_file', type=str, default='routes/Town02_All.txt', help='Filepath for route file')
    parser.add_argument('--model', type=str, default='transformer_av_model.pt', help='Name of saved model')
    args = parser.parse_args()
    
    logging.basicConfig(filename='evaluation.log', 
                        level=logging.INFO,
                        format='%(message)s' ) 

    main(args)