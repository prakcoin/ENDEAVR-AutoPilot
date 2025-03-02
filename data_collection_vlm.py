import argparse
import os
import numpy as np
import carla
from PIL import Image
import json
from utils.vlm_utils import (lidar_to_ego_coordinate, normalize_angle_degree, align_lidar,
                             generate_prompt, generate_label, get_scene_description)
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, 
                                cleanup, update_spectator, read_routes, spawn_pedestrians,
                                cleanup_pedestrians)
from utils.sensors import start_vlm_camera, start_collision_sensor, start_lidar_sensor
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

def run_episode(world, weather, ego_vehicle, agent, rgb_cam, lidar_sensor, end_point, episode, args):
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

    last_lidar = None
    last_ego_transform = None

    frame = 0
    while True:
        if end_episode(ego_vehicle, end_point, frame, args):
            break

        update_spectator(spectator, ego_vehicle)
        
        correct_control, noisy_control, incorrect_control, ec = agent.run_step()
        if noisy_control:
            ego_vehicle.apply_control(noisy_control)

        rgb_data = to_rgb(rgb_cam.get_sensor_data())
        lidar_data = lidar_to_ego_coordinate(lidar_sensor.get_sensor_data())

        img = rgb_cam.get_sensor_data()
        img = np.reshape(np.copy(img.raw_data), (img.height, img.width, 4))

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
        hlc = agent.get_next_action()

        map = world.get_map()
        ego_location = ego_vehicle.get_location()
        waypoint = map.get_waypoint(ego_location)
        correct_finetune_prompt = generate_prompt(hlc, speed_km_h, correct_control.steer, correct_control.brake, correct_control.throttle)
        incorrect_finetune_prompt = generate_prompt(hlc, speed_km_h, incorrect_control.steer, incorrect_control.brake, incorrect_control.throttle)
        scene_description = get_scene_description(world=world, ego_vehicle=ego_vehicle, lidar=lidar_360)
        correct_label = generate_label(world, ego_vehicle, hlc, speed_km_h, correct_control.steer, correct_control.brake, correct_control.throttle, waypoint, ec, scene_description, True)
        incorrect_label = generate_label(world, ego_vehicle, hlc, speed_km_h, correct_control.steer, correct_control.brake, correct_control.throttle, waypoint, ec, scene_description, False, incorrect_control.steer, incorrect_control.brake, incorrect_control.throttle)

        image_filename = f"{args.town}_{weather}_episode_{episode + 1}_frame_{frame:06d}.jpg"
        
        if not agent.noise:
            images.append((image_filename, rgb_data))
            data.append({
                "image": f"{args.image_path}{image_filename}",
                "prompt": correct_finetune_prompt,
                "label": correct_label
            })
            # data.append({
            #     "image": f"{args.image_path}{image_filename}",
            #     "prompt": incorrect_finetune_prompt,
            #     "label": incorrect_label
            # })

        last_ego_transform = ego_vehicle.get_transform()
        last_lidar = lidar_data

        world.tick()
        frame += 1

    if not has_collision and frame <= args.max_frames:
        save_images(images_dir, images)
        save_episode_data(prompts_labels_path, data)

def main(args):
    world, client = init_world(args.town)
    traffic_manager = setup_traffic_manager(client)

    weather_conditions = [
        "ClearNoon", 
        "MidRainSunset", 
        "CloudyNight", 
        "WetSunset", 
        "HardRainNoon", 
        "SoftRainNight",
    ]
    weather_choice = "ClearNoon"
    world.set_weather(getattr(carla.WeatherParameters, weather_choice))
    world.tick()
    route_configs = read_routes(args.route_file)
    episode_count = args.episodes

    all_id, all_actors, vehicle_list = [], [], []
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
        agent = VLMAgent(ego_vehicle, traffic_manager)
        agent.set_route(route, end_point)

        if (args.vehicles > 0):
            vehicle_list = spawn_vehicles(world, client, args.vehicles, traffic_manager, cars_only=False)
        if (args.pedestrians > 0):
            all_id, all_actors, _ = spawn_pedestrians(world, client, args.pedestrians)

        rgb_cam = start_vlm_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        lidar_sensor = start_lidar_sensor(world, ego_vehicle)
        sensors = [rgb_cam.get_sensor(), collision_sensor, lidar_sensor.get_sensor()]
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, weather_choice, ego_vehicle, agent, rgb_cam, lidar_sensor, end_point, episode, args)
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
    parser = argparse.ArgumentParser(description='CARLA Data Collection (VLM) Script')
    parser.add_argument('--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('--max_frames', type=int, default=8000, help='Number of frames to collect per episode')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--pedestrians', type=int, default=40, help='Number of pedestrians present')
    parser.add_argument('--route_file', type=str, default='routes/Town01_VLM.txt', help='Filepath for route file')
    parser.add_argument('--image_path', type=str, default='correct images/', help='Filepath for images')
    args = parser.parse_args()

    main(args)