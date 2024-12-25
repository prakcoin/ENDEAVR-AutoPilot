import os
import argparse
import torch
import carla
import logging
import numpy as np
from PIL import Image
from openai import OpenAI
from utils.sensors import start_camera, start_vlm_camera, start_collision_sensor
from utils.shared_utils import (init_world, read_routes, create_route, traffic_light_to_int, to_depth,
                                spawn_ego_vehicle, spawn_vehicles, setup_traffic_manager, traffic_light_to_int,
                                cleanup, update_spectator, to_rgb, calculate_delta_yaw, spawn_pedestrians, cleanup_pedestrians, 
                                model_control, load_model, inject_vehicle_noise, vlm_inference)
from utils.dist_tracker import DistanceTracker
from utils.hlc_loader import HighLevelCommandLoader

COLLISION_WALKER_PENALTY = 0.5
COLLISION_VEHICLE_PENALTY = 0.6
COLLISION_OTHER_PENALTY = 0.65
RED_LIGHT_PENALTY = 0.7
TIMEOUT_PENALTY = 0.7
WRONG_TURN_PENALTY = 0.7

has_collision = False
collision_type = None
def collision_callback(data):
    global has_collision, collision_type
    collision_type = type(data.other_actor)
    has_collision = True

total_num_vehicle_collisions = 0
total_num_walker_collisions = 0
total_num_other_collisions = 0
total_num_red_light_infractions = 0
total_num_timeouts = 0
total_num_wrong_turns = 0

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

def end_episode(ego_vehicle, end_point, frame, max_frames, turning_infraction):
    global num_timeouts, total_num_timeouts
    done = False
    if end_reached(ego_vehicle, end_point):
        logging.info("Target reached, episode ending")
        done = True
    elif frame >= max_frames:
        logging.info("Maximum frames reached, episode ending")
        num_timeouts += 1
        total_num_timeouts += 1
        done = True
    elif turning_infraction:
        logging.info("Turning infraction, episode ending")
        done = True
    return done

def check_collision(prev_collision):
    global has_collision, collision_type, num_other_collisions, num_vehicle_collisions, num_walker_collisions, total_num_other_collisions, total_num_vehicle_collisions, total_num_walker_collisions
    if has_collision:
        if not prev_collision:
            if collision_type == carla.libcarla.Vehicle:
                num_vehicle_collisions += 1
                total_num_vehicle_collisions += 1
            elif collision_type == carla.libcarla.Walker:
                num_walker_collisions += 1
                total_num_walker_collisions += 1
            else:
                num_other_collisions += 1
                total_num_other_collisions += 1
            prev_collision = True
    else:
        prev_collision = False
    has_collision = False
    collision_type = None
    return prev_collision

def run_episode(world, model, device, ego_vehicle, rgb_cam, vlm_cam, depth_cam, end_point, route, route_length, max_frames, openai_client):
    global has_collision, collision_type, num_other_collisions, num_vehicle_collisions, num_walker_collisions, total_num_other_collisions, total_num_vehicle_collisions, total_num_walker_collisions
    num_other_collisions = 0
    num_vehicle_collisions = 0
    num_walker_collisions = 0
    has_collision = False
    global num_wrong_turns, total_num_wrong_turns
    num_wrong_turns = 0

    global num_red_light_infractions, total_num_red_light_infractions
    num_red_light_infractions = 0

    global num_timeouts, total_num_timeouts
    num_timeouts = 0

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
    running_light = False
    prev_collision = False
    while True:
        prev_collision = check_collision(prev_collision)

        if end_episode(ego_vehicle, end_point, frame, max_frames, turning_infraction):
            break
        
        transform = ego_vehicle.get_transform()
        vehicle_location = transform.location

        velocity = ego_vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_km_h = 3.6 * speed_m_s

        speed_km_h = np.array([speed_km_h])

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
                total_num_wrong_turns += 1
            delta_yaw = 0
        
        prev_hlc = hlc

        update_spectator(spectator, ego_vehicle)
        sensor_data = np.array(to_rgb(rgb_cam.get_sensor_data()))
        depth_map = np.array(to_depth(depth_cam.get_sensor_data()))
        vlm_image = Image.fromarray(to_rgb(vlm_cam.get_sensor_data()))

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
                    total_num_red_light_infractions += 1
            else:
                running_light = False
        light = np.array([traffic_light_to_int(light_status)])

        control = model_control(sensor_data, depth_map, hlc, speed_km_h, light, model, device)
        vlm_control = vlm_inference(openai_client, vlm_image, hlc, speed_km_h, control.steer, control.brake, control.throttle)
        print(vlm_control)
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

    openai_client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=f"http://{args.ip}:{args.port}/v1",
    )

    world, client = init_world(args.town)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))

    traffic_manager = setup_traffic_manager(client)
    route_configs = read_routes(args.route_file)
    episode_count = min(len(route_configs), args.episodes)
    
    all_id, all_actors, vehicle_list = [], [], [], []
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
            inject_vehicle_noise(world, vehicle_list, traffic_manager)
        if (args.pedestrians > 0):
            all_id, all_actors, _ = spawn_pedestrians(world, client, args.pedestrians)


        rgb_cam, depth_cam = start_camera(world, ego_vehicle)
        vlm_cam = start_vlm_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors = [rgb_cam.get_sensor(), vlm_cam.get_sensor(), depth_cam.get_sensor(), collision_sensor]

        episode_completed, route_completion = run_episode(world, model, device, ego_vehicle, rgb_cam, vlm_cam, depth_cam, end_point, route, route_length, args.max_frames, openai_client)
        if episode_completed:
            completed_episodes += 1

        route_completions.append(route_completion)
        infraction_penalty = COLLISION_VEHICLE_PENALTY ** num_vehicle_collisions * \
                            COLLISION_WALKER_PENALTY ** num_walker_collisions * \
                            COLLISION_OTHER_PENALTY ** num_other_collisions * \
                            RED_LIGHT_PENALTY ** num_red_light_infractions * \
                            TIMEOUT_PENALTY ** num_timeouts * \
                            WRONG_TURN_PENALTY ** num_wrong_turns
        infraction_penalties.append(infraction_penalty)
        driving_score = infraction_penalty * route_completion
        driving_scores.append(driving_score)
        logging.info(f"Infraction penalty: {infraction_penalty}")
        logging.info(
            f"Infraction Breakdown:\n"
            f"  - Vehicle Collisions: {num_vehicle_collisions}\n"
            f"  - Walker Collisions: {num_walker_collisions}\n"
            f"  - Other Collisions: {num_other_collisions}\n"
            f"  - Red Light Infractions: {num_red_light_infractions}\n"
            f"  - Timeouts: {num_timeouts}\n"
            f"  - Wrong Turns: {num_wrong_turns}\n"
            f"Total Infractions: {num_vehicle_collisions + num_walker_collisions + num_other_collisions + num_red_light_infractions + num_timeouts + num_wrong_turns}"
        )
        logging.info(f"Driving score: {driving_score}")
        cleanup(client, ego_vehicle, vehicle_list, sensors)
        cleanup_pedestrians(client, all_id, all_actors)

    logging.info(f"Episode completion rate: {completed_episodes / episode_count}")
    logging.info(f"Average route completion: {sum(route_completions) / episode_count}")
    logging.info(f"Average infraction penalty: {sum(infraction_penalties) / episode_count}")
    logging.info(
        f"Infraction Breakdown:\n"
        f"  - Vehicle Collisions: {total_num_vehicle_collisions}\n"
        f"  - Walker Collisions: {total_num_walker_collisions}\n"
        f"  - Other Collisions: {total_num_other_collisions}\n"
        f"  - Red Light Infractions: {total_num_red_light_infractions}\n"
        f"  - Timeouts: {total_num_timeouts}\n"
        f"  - Wrong Turns: {total_num_wrong_turns}\n"
        f"Total Infractions: {total_num_vehicle_collisions + total_num_walker_collisions + total_num_other_collisions + total_num_red_light_infractions + total_num_timeouts + total_num_wrong_turns}"
    )
    logging.info(f"Average driving score: {sum(driving_scores) / episode_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Model Evaluation Script')
    parser.add_argument('--town', type=str, default='Town02', help='CARLA town to use')
    parser.add_argument('--weather', type=str, default='ClearNoon', help='Weather condition to set')
    parser.add_argument('--max_frames', type=int, default=5000, help='Number of frames before terminating episode')
    parser.add_argument('--episodes', type=int, default=12, help='Number of episodes to evaluate for')
    parser.add_argument('--vehicles', type=int, default=50, help='Number of vehicles present')
    parser.add_argument('--pedestrians', type=int, default=50, help='Number of pedestrians present')
    parser.add_argument('--route_file', type=str, default='routes/Town02_All.txt', help='Filepath for route file')
    parser.add_argument('--model', type=str, default='av_model.pt', help='Name of saved model')
    parser.add_argument('--ip', type=str, default='localhost', help='IP address of VLM server')
    parser.add_argument('--port', type=str, default='8000', help='Port of VLM server')
    args = parser.parse_args()
    
    logging.basicConfig(filename=f'evaluation_{args.model}.log', 
                        level=logging.INFO,
                        format='%(message)s' ) 

    main(args)