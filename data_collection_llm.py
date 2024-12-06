import argparse
import os
import numpy as np
import carla
from datasets import Dataset
from PIL import Image
import json
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, 
                                road_option_to_int, cleanup, update_spectator, read_routes, 
                                get_traffic_light_status, traffic_light_to_int)
from utils.sensors import start_vlm_camera, start_collision_sensor
from utils.agents import LLMAgent

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

def generate_prompt(hlc, speed, light, steer, brake, throttle):    
    prompt = (
        "You are a powerful vehicle control assistant. Analyze the following sensor data from a front RGB camera in the CARLA Simulator along with additional context data. "
        "The control signals given below were generated by an end-to-end convolutional neural network.\n\n"
        f"- Current high-level command (HLC): {hlc} ('0' = follow lane, '1' = left turn at intersection, '2' = right turn at intersection, '3' = straight at intersection).\n"
        f"- Current speed: {speed:.3f} km/h. Current traffic light status: {light} (0 = vehicle not at light, 1 = red, 2 = green, 3 = yellow).\n\n"
        "The predicted control signals generated from the end-to-end convolutional neural network are as follows:\n\n"
        f"- Steering angle: {steer:.3f} (range: -1.0 to 1.0, where positive values indicate a right turn and negative values indicate a left turn)\n"
        f"- Brake value: {brake:.3f} (range: 0.0 to 1.0, where 0.0 is no braking (no deceleration), and 1.0 is full braking (vehicle comes to a stop))\n"
        f"- Throttle value: {throttle:.3f} (range: 0.0 to 1.0, where 0.0 is no acceleration (vehicle not moving), and 1.0 is full acceleration (vehicle at maximum speed))\n\n"
        "Determine if the predicted control signals are correct. If correct, confirm them. If incorrect, provide the appropriate values for safe and efficient vehicle control."
    )
    return prompt

def generate_label(steer, brake, throttle):
    label = (
        f"The appropriate control signals are:\n\n"
        f"- Steering Angle: {steer:.3f}\n"
        f"- Brake: {brake:.3f}\n"
        f"- Throttle: {throttle:.3f}"
    )
    return label

def save_episode_data(prompts_labels_path, episode_data):
    if os.path.exists(prompts_labels_path):
        with open(prompts_labels_path, "r") as f:
            all_data = json.load(f)
    else:
        all_data = []

    all_data.extend(episode_data)

    with open(prompts_labels_path, "w") as f:
        json.dump(all_data, f, indent=4)

def run_episode(world, ego_vehicle, agent, rgb_cam, end_point, collect_correct, episode, args):
    global has_collision
    has_collision = False

    images_dir = os.path.join("llm_data", "images")
    os.makedirs(images_dir, exist_ok=True)
    prompts_labels_path = os.path.join("llm_data", "prompts_labels.json")

    data = []

    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    frame = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, args):
            break

        update_spectator(spectator, ego_vehicle)
        
        correct_control, incorrect_control = agent.run_step()
        ego_vehicle.apply_control(correct_control)

        rgb_data = to_rgb(rgb_cam.get_sensor_data())
        
        velocity = ego_vehicle.get_velocity()
        speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        hlc = road_option_to_int(agent.get_next_action())
        light = traffic_light_to_int(get_traffic_light_status(ego_vehicle))
        
        selected_control = correct_control if collect_correct else incorrect_control
        finetune_prompt = generate_prompt(hlc, speed_km_h, light, selected_control.steer, selected_control.brake, selected_control.throttle)
        label = generate_label(correct_control.steer, correct_control.brake, correct_control.throttle)
        
        correct_str = "correct" if collect_correct else "incorrect"
        image_filename = f"{args.town}_episode_{episode + 1}_{correct_str}_frame_{frame:06d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        image = Image.fromarray(rgb_data)
        image.save(image_path)

        data.append({
            "image": image_filename,
            "prompt": finetune_prompt,
            "label": label
        })

        world.tick()
        frame += 1

    save_episode_data(prompts_labels_path, data)

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
    collect_correct = True
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
        agent = LLMAgent(ego_vehicle, traffic_manager)
        agent.set_route(route, end_point)

        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager)

        rgb_cam = start_vlm_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors = [rgb_cam.get_sensor(), collision_sensor]
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, ego_vehicle, agent, rgb_cam, end_point, collect_correct, episode, args)
        if (has_collision):
            num_tries += 1
            episode -= 1
            restart = True
            print("Redoing ", end="")
        else:
            collect_correct = not collect_correct
            restart = False
        cleanup(client, ego_vehicle, vehicle_list, sensors)
        episode += 1
    print("Simulation complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection (LLM) Script')
    parser.add_argument('--town', type=str, default='Town02', help='CARLA town to use')
    parser.add_argument('--weather', type=str, default='ClearNoon', help='CARLA weather conditions to use')
    parser.add_argument('--max_frames', type=int, default=8000, help='Number of frames to collect per episode')
    parser.add_argument('--episodes', type=int, default=16, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--route_file', type=str, default='routes/Town02_All.txt', help='Filepath for route file')
    args = parser.parse_args()

    main(args)