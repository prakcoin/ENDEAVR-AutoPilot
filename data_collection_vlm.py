import argparse
import os
import random
import numpy as np
import carla
from PIL import Image
import json
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, 
                                cleanup, update_spectator, read_routes, spawn_pedestrians,
                                cleanup_pedestrians, get_traffic_light_status)
from utils.sensors import start_vlm_camera, start_collision_sensor, start_semantic_segmentation_sensor
from utils.agents import VLMAgent

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

def generate_prompt(hlc, speed, steer, brake, throttle):  
    road_option_dict = {
        "LaneFollow": "Follow the lane",
        "Left": "Turn left at the junction",
        "Right": "Turn right at the junction",
        "Straight": "Go straight at the junction"
    }    
    lang_hlc = road_option_dict[hlc]

    prompt = (
        "Analyze the following sensor data along with additional context data.\n\n"
        f"- Current high-level command: {lang_hlc}\n"
        f"- Current speed: {speed:.3f} km/h\n"
        f"- Predicted steer value: {steer:.3f}\n"
        f"- Predicted brake value: {brake:.3f}\n"
        f"- Predicted throttle value: {throttle:.3f}\n\n"
        "Determine if the predicted control signals are correct. If correct, confirm them. If incorrect, provide the appropriate values for safe vehicle control."
    )
    return prompt

def generate_label(weather, steer, brake, throttle, light, waypoint, lang_error):
    light_dict = {
        -1: "The ego vehicle is not at a traffic light.",
        carla.libcarla.TrafficLightState.Red: "The traffic light is red.",
        carla.libcarla.TrafficLightState.Green: "The traffic light is green.",
        carla.libcarla.TrafficLightState.Yellow: "The traffic light is yellow."
    }

    weather_conditions = {
        "ClearNoon": "The weather is clear and sunny at noon.",
        "ClearSunset": "The weather is clear at sunset.",
        "ClearNight": "The weather is clear at night.",
        "CloudyNoon": "The weather is cloudy at noon.",
        "CloudySunset": "The weather is cloudy at sunset.",
        "CloudyNight": "The weather is cloudy at night.",
        "SoftRainNoon": "It is softly raining at noon.",
        "SoftRainSunset": "It is softly raining at sunset.",
        "SoftRainNight": "It is softly raining at night.",
        "WetCloudyNoon": "It is wet and cloudy at noon.",
        "WetCloudySunset": "It is wet and cloudy at sunset.",
        "WetCloudyNight": "It is wet and cloudy at night.",
        "HardRainNoon": "It is heavily raining at noon.",
        "HardRainSunset": "It is heavily raining at sunset.",
        "HardRainNight": "It is heavily raining at night.",
        "DustStorm": "There is a dust storm."
    }
    
    lang_weather = weather_conditions[weather]
    lang_light = light_dict[light]
    at_junction = waypoint.is_junction
    
    if at_junction:
        road_description = "The ego vehicle is at a junction."
    else:
        lane_type = waypoint.lane_type.name.lower()
        left_lane_marking = waypoint.left_lane_marking.type.name.lower()
        
        road_description = (
            f"The ego vehicle isn't at a junction. The road is a {lane_type} road with a {left_lane_marking} left lane marking."
        )

    label = (
        f"{lang_weather} {lang_light} {road_description} {lang_error}\n"
        f"Therefore, the appropriate control signals are:\n\n"
        f"- Steering Angle: {steer:.3f}\n"
        f"- Brake: {brake:.3f}\n"
        f"- Throttle: {throttle:.3f}"
    )
    return label

def save_images(images_dir, images):
    for image_filename, rgb_data in images:
        image_path = os.path.join(images_dir, image_filename)
        image = Image.fromarray(rgb_data)
        image.save(image_path)

def save_episode_data(prompts_labels_path, episode_data):
    if os.path.exists(prompts_labels_path):
        with open(prompts_labels_path, "r") as f:
            all_data = json.load(f)
    else:
        all_data = []

    all_data.extend(episode_data)

    with open(prompts_labels_path, "w") as f:
        json.dump(all_data, f, indent=4)

def run_episode(world, weather, ego_vehicle, agent, rgb_cam, end_point, collect_correct, episode, args):
    global has_collision
    has_collision = False

    images_dir = os.path.join("vlm_data", "images")
    os.makedirs(images_dir, exist_ok=True)
    prompts_labels_path = os.path.join("vlm_data", "prompts_labels.json")

    data = []
    images = []

    spectator = world.get_spectator()
    for _ in range(10):
        world.tick()

    frame = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, args):
            break

        update_spectator(spectator, ego_vehicle)
        
        correct_control, incorrect_control, lang_error = agent.run_step()
        lang_error = "The predicted control signals are correct." if collect_correct else lang_error
        ego_vehicle.apply_control(correct_control)

        rgb_data = to_rgb(rgb_cam.get_sensor_data())
        
        velocity = ego_vehicle.get_velocity()
        speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        hlc = agent.get_next_action()
        light = get_traffic_light_status(ego_vehicle)

        map = world.get_map()
        ego_location = ego_vehicle.get_location()
        waypoint = map.get_waypoint(ego_location)
        selected_control = correct_control if collect_correct else incorrect_control
        finetune_prompt = generate_prompt(hlc, speed_km_h, selected_control.steer, selected_control.brake, selected_control.throttle)
        label = generate_label(weather, correct_control.steer, correct_control.brake, correct_control.throttle, light, waypoint, lang_error)

        correct_str = "correct" if collect_correct else "incorrect"
        image_filename = f"{args.town}_episode_{episode + 1}_{correct_str}_frame_{frame:06d}.jpg"
        images.append((image_filename, rgb_data))

        data.append({
            "image": f"{args.image_path}{image_filename}",
            "prompt": finetune_prompt,
            "label": label
        })

        world.tick()
        frame += 1

    if not has_collision and frame <= args.max_frames:
        save_images(images_dir, images)
        save_episode_data(prompts_labels_path, data)

def main(args):
    world, client = init_world(args.town)
    traffic_manager = setup_traffic_manager(client)

    weather_conditions = ["ClearNoon", "ClearSunset", "ClearNight", "CloudyNoon", 
                          "CloudySunset", "CloudyNight", "SoftRainNoon", "SoftRainSunset",
                          "SoftRainNight", "WetCloudyNoon", "WetCloudySunset", "WetCloudyNight", 
                          "HardRainNoon", "HardRainSunset", "HardRainNight", "DustStorm"]

    route_configs = read_routes(args.route_file)
    episode_count = args.episodes

    all_id, all_actors, vehicle_list = [], [], []
    restart = False
    episode = 0
    collect_correct = True
    while episode < episode_count:
        print(f'Episode: {episode + 1}')
        if not restart:
            weather_choice = random.choice(weather_conditions)
            weather_conditions.remove(weather_choice)
            world.set_weather(getattr(carla.WeatherParameters, weather_choice))
            world.tick()

            num_tries = 0
            spawn_point_index, end_point_index, _, route = create_route(route_configs)
        
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[spawn_point_index]
        end_point = spawn_points[end_point_index]

        print(f"Route from spawn point #{spawn_point_index} to #{end_point_index}")

        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        agent = VLMAgent(ego_vehicle, traffic_manager)
        agent.set_route(route, end_point)

        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager, cars_only=False)
        if (args.pedestrians > 0):
            all_id, all_actors, _ = spawn_pedestrians(world, client, args.pedestrians)

        rgb_cam = start_vlm_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors = [rgb_cam.get_sensor(), collision_sensor]
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, weather_choice, ego_vehicle, agent, rgb_cam, end_point, collect_correct, episode, args)
        if (has_collision):
            num_tries += 1
            episode -= 1
            restart = True
            print("Redoing ", end="")
        else:
            collect_correct = not collect_correct
            restart = False
        cleanup(client, ego_vehicle, vehicle_list, sensors)
        cleanup_pedestrians(client, all_id, all_actors)
        episode += 1
    print("Simulation complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection (VLM) Script')
    parser.add_argument('--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('--max_frames', type=int, default=8000, help='Number of frames to collect per episode')
    parser.add_argument('--episodes', type=int, default=16, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--pedestrians', type=int, default=40, help='Number of pedestrians present')
    parser.add_argument('--route_file', type=str, default='routes/Town01_Train.txt', help='Filepath for route file')
    parser.add_argument('--image_path', type=str, default='vlm_data/images/', help='Filepath for images')
    args = parser.parse_args()

    main(args)