import argparse
import os
import numpy as np
import carla
from PIL import Image
import json
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, to_depth,
                                road_option_to_int, cleanup, update_spectator, read_routes, 
                                get_traffic_light_status, traffic_light_to_int)
from utils.sensors import start_camera, start_collision_sensor
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
    return done

def save_image_from_array(image_array, save_path):
    if not os.path.isdir(f'data/images'):
        os.makedirs(f'data/images')
    image = Image.fromarray(image_array)
    image.save(f"data/images/{save_path}", format='JPEG')

def append_jsonl_entry(jsonl_file, entry):
    if not os.path.isdir(f'data'):
        os.makedirs(f'data')
    with open(f'data/{jsonl_file}', 'a') as f:
        json.dump(entry, f)
        f.write('\n')

def generate_annotation_prompt(hlc, speed, light, steer, brake, throttle, correct_steer=None, correct_brake=None, correct_throttle=None):
    prompt = (
        "Analyze the following sensor data from a front RGB camera in the CARLA Simulator along with additional context data.\n\n"
        f"Current high-level command (HLC): {hlc} (0 = follow lane, 1 = left turn at intersection, 2 = right turn at intersection, 3 = straight at intersection).\n"
        f"Current speed: {speed:.3f} km/h. Current traffic light status: {light} (0 = vehicle not at light, 1 = red, 2 = green, 3 = yellow).\n\n"
        "The control signals generated by the end-to-end convolutional neural network are as follows:\n\n"
        f"- Steering angle: {steer:.3f} (range: -1.0 to 1.0, where positive values indicate a right turn and negative values indicate a left turn)\n"
        f"- Brake value: {brake:.3f} (range: 0.0 to 1.0, where 0.0 is no braking (no deceleration), and 1.0 is full braking (vehicle comes to a stop))\n"
        f"- Throttle value: {throttle:.3f} (range: 0.0 to 1.0, where 0.0 is no acceleration (vehicle not moving), and 1.0 is full acceleration (vehicle at maximum speed))\n\n"
    )
    
    if correct_steer is not None and correct_brake is not None and correct_throttle is not None:
        prompt += (
            "The provided control signals appear to be incorrect. Your task is to assess the original control signals first, "
            "based on the sensor data and the driving context provided, and explain why they are inappropriate for the given situation.\n\n"
            "Once you've assessed the issue with the provided signals, justify why the corrected control values are more suitable for the current scenario.\n\n"
            "The correct control signals for this situation are as follows:\n\n"
            f"- Correct Steering angle: {correct_steer:.3f}\n"
            f"- Correct Brake value: {correct_brake:.3f}\n"
            f"- Correct Throttle value: {correct_throttle:.3f}\n\n"
            "Please provide your response in the following format:\n\n"
            "Explanation:\n"
            "[Describe why the original control signals are inappropriate and why the corrected values align better with the sensor data and context. "
            "Avoid list format, and keep your reasoning directly tied to the data presented.]"
        )

    else:
        prompt += (
            "Your task is to confirm that the control signals are appropriate for the given driving situation. "
            "Please explain why the provided control signals are correct, based on the sensor data and context. "
            "Ensure your reasoning is directly tied to the provided information and the current driving scenario.\n\n"
            "Please provide your response in the following format:\n\n"
            "Explanation:\n"
            "[Describe why the control signals are appropriate for the driving scenario based on the sensor data and context. "
            "Avoid list format, and keep your reasoning directly tied to the data presented.]"
        )

    
    return prompt


def generate_prompt(hlc, speed, light, steer, brake, throttle):    
    prompt = (
        "You are a powerful vehicle control assistant. Analyze the following sensor data from a front RGB camera in the CARLA Simulator along with additional context data. "
        "The control signals given below were generated by an end-to-end convolutional neural network.\n\n"
        "<image>\n\n"
        f"Current high-level command (HLC): {hlc} ('0' = follow lane, '1' = left turn at intersection, '2' = right turn at intersection, '3' = straight at intersection). "
        f"Current speed: {speed:.3f} km/h. Current traffic light status: {light} (0 = vehicle not at light, 1 = red, 2 = green, 3 = yellow). "
        "The predicted control signals generated from the end-to-end convolutional neural network are as follows:\n\n"
        f"- Steering angle: {steer:.3f} (range: -1.0 to 1.0, where positive values indicate a right turn and negative values indicate a left turn)\n"
        f"- Brake value: {brake:.3f} (range: 0.0 to 1.0, where 0.0 is no braking (no deceleration), and 1.0 is full braking (vehicle comes to a stop))\n"
        f"- Throttle value: {throttle:.3f} (range: 0.0 to 1.0, where 0.0 is no acceleration (vehicle not moving), and 1.0 is full acceleration (vehicle at maximum speed))\n\n"
        "Your task is to evaluate whether these control signals are appropriate for the given driving situation based on the sensor data. "
        "If the control signals seem inaccurate or unsafe, adjust them accordingly and explain your corrections. "
        "Otherwise, confirm the control signals as correct and explain why they are appropriate.\n\n"
        "Please provide your response in the following format:\n\n"
        "Control values:\nSteering Angle: [your_value], Brake: [your_value], Throttle: [your_value]\n\n"
        "Explanation:\n[Provide a clear explanation of why the control signals are appropriate or why adjustments were necessary, based on the sensor data and context.]"
    )
    
    return prompt


def run_episode(world, ego_vehicle, agent, rgb_cam, end_point, args):
    global has_collision
    has_collision = False

    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    frame = 0
    idle_frames = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, idle_frames, args):
            break

        update_spectator(spectator, ego_vehicle)
        
        correct_control, incorrect_control = agent.run_step()
        ego_vehicle.apply_control(correct_control)

        rgb_data = to_rgb(rgb_cam.get_sensor_data())
        
        velocity = ego_vehicle.get_velocity()
        speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        hlc = road_option_to_int(agent.get_next_action())
        light = traffic_light_to_int(get_traffic_light_status(ego_vehicle))

        if speed_km_h == 0.0:
            idle_frames += 1
        else:
            idle_frames = 0

        finetune_prompt = generate_prompt(hlc, speed_km_h, light, correct_control.steer, correct_control.brake, correct_control.throttle)
        annotation_prompt = generate_annotation_prompt(hlc, speed_km_h, light, correct_control.steer, correct_control.brake, correct_control.throttle)
        image_path = f'correct_frame_{frame:05d}.jpg'
        entry_id = f'correct_frame_{frame:05d}'
        if args.incorrect:
            finetune_prompt = generate_prompt(hlc, speed_km_h, light, incorrect_control.steer, incorrect_control.brake, incorrect_control.throttle)
            annotation_prompt = generate_annotation_prompt(hlc, speed_km_h, light, incorrect_control.steer, incorrect_control.brake, incorrect_control.throttle, correct_control.steer, correct_control.brake, correct_control.throttle)
            image_path = f'incorrect_frame_{frame:05d}.jpg'
            entry_id = f'incorrect_frame_{frame:05d}'

        save_image_from_array(np.array(rgb_data), image_path)

        annotation_json_entry = {
            "image": f"Images/{image_path}",
            "prompt": annotation_prompt
        }

        finetune_jsonl_entry = {
            "id": entry_id,
            "image": f"Images/{image_path}",
            "width": 320,
            "height": 240,
            "conversations": [
                {
                    "from": "human",
                    "value": finetune_prompt
                },
                {
                    "from": "gpt",
                    "value": (
                        f"Control values:\n"
                        f"Steering Angle: {correct_control.steer:.3f}, Brake: {correct_control.brake:.3f}, Throttle: {correct_control.throttle:.3f}\n\n"
                        f"Explanation:\n"
                    )
                }
            ]
        }
            
        append_jsonl_entry('annotation_prompts.jsonl', annotation_json_entry)
        append_jsonl_entry('complete_dataset.jsonl', finetune_jsonl_entry)
        world.tick()
        frame += 1
    print(f"Total frames: {frame}")

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
        agent = LLMAgent(ego_vehicle, traffic_manager)
        agent.set_route(route, end_point)

        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager)

        rgb_cam, _ = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors = [rgb_cam.get_sensor(), collision_sensor]
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, ego_vehicle, agent, rgb_cam, end_point, args)
        if (has_collision):
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
    parser = argparse.ArgumentParser(description='CARLA Data Collection (LLM) Script')
    parser.add_argument('--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('--weather', type=str, default='ClearNoon', help='CARLA weather conditions to use')
    parser.add_argument('--max_frames', type=int, default=8000, help='Number of frames to collect per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--route_file', type=str, default='routes/Town01_LLM.txt', help='Filepath for route file')
    parser.add_argument('--incorrect', action="store_true", help='Collect incorrect control signals')
    args = parser.parse_args()

    # logging.basicConfig(filename='data_collection_log.log', 
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s' ) 

    main(args)