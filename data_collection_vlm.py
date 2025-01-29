import argparse
import os
import random
import numpy as np
import carla
from PIL import Image
import json
import cv2
from scipy.spatial import KDTree
import webcolors
from utils.shared_utils import (init_world, setup_traffic_manager, setup_vehicle_for_tm, 
                                spawn_ego_vehicle, spawn_vehicles, create_route, to_rgb, 
                                cleanup, update_spectator, read_routes, spawn_pedestrians,
                                cleanup_pedestrians, get_traffic_light_status)
from utils.sensors import start_vlm_camera, start_collision_sensor
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

def convert_rgb_to_names(rgb_tuple):
    css3_db = webcolors.CSS2_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(webcolors.hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    _, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}'

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

def generate_scene_description(scene_description):
    descriptions = []
    vehicle_count, ped_count = 0, 0

    for obj in scene_description:
        if obj["type"] == "vehicle":
            vehicle_count += 1
        elif obj["type"] == "pedestrian":
            ped_count += 1

    descriptions.append(f"There are {vehicle_count} vehicles and {ped_count} pedestrians nearby.")

    for obj in scene_description:
        distance = obj["distance"]
        if distance < 10:
            proximity_str = "very close"
        elif distance < 20:
            proximity_str = "close"
        elif distance < 35:
            proximity_str = "at a moderate distance"
        else:
            proximity_str = "far away"

        if obj["type"] == "vehicle":
            if -2 <= obj['position'][1] <= 2:
                rough_pos_str = 'directly in front of the ego vehicle'
            elif obj['position'][1] > 2:
                rough_pos_str = 'to the front right of the ego vehicle'
            else:
                rough_pos_str = 'to the front left of the ego vehicle'

            if obj["speed"] < 0.2:
                motion_status = "stopped"
            elif obj["speed"] < 5:
                motion_status = "moving slowly"
            else:
                motion_status = "moving"

            steer = obj.get("steer", 0)
            if steer < -0.1:
                turning_status = "turning left"
            elif steer < -0.03:
                turning_status = "turning slightly left"
            elif steer > 0.1:
                turning_status = "turning right"
            elif steer > 0.03:
                turning_status = "turning slightly right"
            else:
                turning_status = "going straight"

            orientation_relative_to_ego = obj.get('yaw', 0) * 180 / np.pi 
            if -135 < orientation_relative_to_ego < -45:
                orientation_str = 'pointing leftwards'
            elif 45 < orientation_relative_to_ego < 135:
                orientation_str = 'pointing rightwards'
            elif 135 < orientation_relative_to_ego or orientation_relative_to_ego < -135:
                orientation_str = 'pointing towards the ego vehicle'
            else:
                orientation_str = 'pointing in the same direction as the ego vehicle'

            if 'firetruck' in obj['id']:
                vehicle_type = 'firetruck'
            elif 'police' in obj['id']:
                vehicle_type = 'police car'
            elif 'ambulance' in obj['id']:
                vehicle_type = 'ambulance'
            elif 'jeep' in obj['id']:
                vehicle_type = 'jeep'
            elif 'micro' in obj['id']:
                vehicle_type = 'small car'
            elif 'nissan.patrol' in obj['id']:
                vehicle_type = 'SUV'
            elif 'european_hgv' in obj['id']:
                vehicle_type = 'HGV'
            elif 'sprinter' in obj['id']:
                vehicle_type = 'sprinter'
            else:
                vehicle_type = obj['base_type']

            desc = f"A {obj['color'].lower()} {vehicle_type.lower()}" if obj['color'] else f"A {vehicle_type}"
            desc += f" is {motion_status}, {turning_status}, {orientation_str}, located {rough_pos_str}, and is {proximity_str}."
            descriptions.append(desc)

        elif obj["type"] == "pedestrian":
            if -2 < obj['position'][1] < 2:
                rough_pos_str = 'directly in front of the ego vehicle'
            elif obj['position'][1] > 2:
                rough_pos_str = 'to the front right of the ego vehicle'
            else:
                rough_pos_str = 'to the front left of the ego vehicle'

            if obj["speed"] < 0.2:
                motion_status = "standing"
            else:
                motion_status = "walking"

            desc = f"A pedestrian is {motion_status}, located {rough_pos_str}, and is {proximity_str}."
            descriptions.append(desc)

    return " ".join(descriptions)

def generate_label(weather, correct_steer, correct_brake, correct_throttle, light, waypoint, ec, scene_description, collect_correct):
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

    lang_scene = generate_scene_description(scene_description)
    
    explanation = "Based on the current scene, "
    if not collect_correct:
        
        if ec == "plus_right_steer":
            explanation += "the model predicted excessive rightward steering."
        elif ec == "plus_left_steer":
            explanation += "the model predicted excessive leftward steering."
        elif ec == "plus_throttle":
            explanation += "the model predicted excessive acceleration."
        elif ec == "minus_throttle":
            explanation += "the model predicted insufficient acceleration."
        elif ec == "plus_brake":
            explanation += "the model predicted excessive braking."
        elif ec == "minus_brake":
            explanation += "the model predicted insufficient braking."
        elif ec == "swap_throttle":
            explanation += "the model predicted braking instead of acceleration."
        elif ec == "swap_brake":
            explanation += "the model predicted acceleration instead of braking."
    else:
        explanation += "the predicted control signals are correct."

    lang_weather = weather_conditions[weather]
    lang_light = light_dict[light]
    at_junction = waypoint.is_junction
    
    if at_junction:
        road_description = "The ego vehicle is at a junction."
    else:
        lane_type = waypoint.lane_type.name.lower()
        left_lane_marking = waypoint.left_lane_marking.type.name.lower()
        
        road_description = (
            f"The ego vehicle isn't at a junction, and the road is a {lane_type} road with a {left_lane_marking} left lane marking."
        )

    label = (
        f"{lang_weather} {lang_light} {road_description} {lang_scene} {explanation} "
        f"Therefore, the appropriate control signals are:\n\n"
        f"- Steer: {correct_steer:.3f}\n"
        f"- Brake: {correct_brake:.3f}\n"
        f"- Throttle: {correct_throttle:.3f}"
    )
    return label

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def point_in_canvas(pos, img_h, img_w):
    return 0 <= pos[0] < img_w and 0 <= pos[1] < img_h

def is_object_in_front_of_camera(ray, forward_vec, camera_location, npc, world, fov_angle=60):
    ray_np = np.array([ray.x, ray.y, ray.z])
    forward_vec_np = np.array([forward_vec.x, forward_vec.y, forward_vec.z])

    magnitude_ray = np.linalg.norm(ray_np)
    magnitude_forward_vec = np.linalg.norm(forward_vec_np)
    if magnitude_ray == 0 or magnitude_forward_vec == 0:
        return False

    dot_product = np.dot(ray_np, forward_vec_np)
    angle = np.arccos(np.clip(dot_product / (magnitude_ray * magnitude_forward_vec), -1.0, 1.0))
    angle_deg = np.degrees(angle)

    if angle_deg > fov_angle:
        return False

    start_location = camera_location
    raycast_hits = []

    for edge in npc.bounding_box.get_world_vertices(npc.get_transform()):
        raycast_result = world.cast_ray(start_location, edge)
        raycast_hits.append(raycast_result)

    actor_labels = {carla.libcarla.CityObjectLabel.Car, carla.libcarla.CityObjectLabel.Bus, carla.libcarla.CityObjectLabel.Truck, 
                    carla.libcarla.CityObjectLabel.Motorcycle, carla.libcarla.CityObjectLabel.Bicycle, carla.libcarla.CityObjectLabel.Train,
                    carla.libcarla.CityObjectLabel.Pedestrians}

    other_labels = {carla.libcarla.CityObjectLabel.NONE, carla.libcarla.CityObjectLabel.Roads, carla.libcarla.CityObjectLabel.Poles, 
                    carla.libcarla.CityObjectLabel.TrafficLight, carla.libcarla.CityObjectLabel.TrafficSigns, carla.libcarla.CityObjectLabel.GuardRail,
                    carla.libcarla.CityObjectLabel.Vegetation}

    for hit_list in raycast_hits:
        for hit in hit_list:
            if hit.label is None or hit.label in other_labels:
                continue

            if hit.label in actor_labels:
                return True
            else:
                return False

    return True

def get_relative_transform(ego_matrix, vehicle_matrix):
  relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
  rot = ego_matrix[:3, :3].T
  relative_pos = rot @ relative_pos

  return relative_pos

def normalize_angle(x):
  x = x % (2 * np.pi)
  if x > np.pi:
    x -= 2 * np.pi
  return x

def get_scene_description_and_bounding_boxes(world, vehicle, camera, image, image_h, image_w, K, K_b, display_bb=False):
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    actors = list(world.get_actors().filter("*vehicle*")) + list(world.get_actors().filter("*walker*"))

    scene_description = []
    for npc in actors:
        if npc.id != vehicle.id:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

            if dist < 50:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                is_behind = forward_vec.dot(ray) < 0
                projection_matrix = K_b if is_behind else K

                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                projected_verts = [get_image_point(v, projection_matrix, world_2_camera) for v in verts]

                if is_object_in_front_of_camera(ray, forward_vec, camera.get_transform().location, npc, world):
                    if any(point_in_canvas(v, image_h, image_w) for v in projected_verts):
                        actor_type = "pedestrian" if "walker" in npc.type_id else "vehicle"
                        map = world.get_map()
                        ego_wp = map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
                        ego_matrix = np.array(vehicle.get_transform().get_matrix())
                        ego_rotation = vehicle.get_transform().rotation
                        ego_yaw = np.deg2rad(ego_rotation.yaw)
                        ego_lane_direction = ego_wp.lane_id / abs(ego_wp.lane_id)


                        if (actor_type == "vehicle"):
                            base_type = npc.attributes['base_type']
                            vehicle_wp = map.get_waypoint(npc.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
                            vehicle_control = npc.get_control()
                            vehicle_rotation = npc.get_transform().rotation
                            vehicle_matrix = np.array(npc.get_transform().get_matrix())
                            relative_pos = get_relative_transform(ego_matrix, vehicle_matrix)
                            same_road_as_ego = False
                            same_direction_as_ego = False
                            direction = vehicle_wp.lane_id / abs(vehicle_wp.lane_id)
                            speed = (3.6 * np.sqrt(npc.get_velocity().x**2 + npc.get_velocity().y**2 + npc.get_velocity().z**2))
                            yaw = np.deg2rad(vehicle_rotation.yaw)
                            relative_yaw = normalize_angle(yaw - ego_yaw)
                            if direction == ego_lane_direction:
                                same_direction_as_ego = True
                            if vehicle_wp.road_id == ego_wp.road_id:
                                same_road_as_ego = True
                            try:
                                rgb = tuple(map(int, npc.attributes['color'].split(',')))
                                color_name = convert_rgb_to_names(rgb)
                            except:
                                rgb = None
                                color_name = None

                            scene_description.append({
                                "type": actor_type,
                                "base_type": base_type,
                                "same_road": same_road_as_ego,
                                "same_dir": same_direction_as_ego,
                                'throttle': vehicle_control.throttle,
                                'brake': vehicle_control.brake,
                                'steer': vehicle_control.steer,
                                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                                "is_in_junction": vehicle_wp.is_junction,
                                "junction_id": vehicle_wp.junction_id,
                                'yaw': relative_yaw,
                                "speed": speed,
                                "color": color_name,
                                "id": npc.type_id,
                                "distance": dist
                            })
                        else:
                            ped_wp = map.get_waypoint(npc.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
                            ped_matrix = np.array(npc.get_transform().get_matrix())
                            relative_pos = get_relative_transform(ego_matrix, ped_matrix)
                            same_road_as_ego = False
                            same_direction_as_ego = False
                            direction = ped_wp.lane_id / abs(ped_wp.lane_id)
                            speed = (3.6 * np.sqrt(npc.get_velocity().x**2 + npc.get_velocity().y**2 + npc.get_velocity().z**2))
                            if direction == ego_lane_direction:
                                same_direction_as_ego = True
                            if ped_wp.road_id == ego_wp.road_id:
                                same_road_as_ego = True
                            scene_description.append({
                                "type": actor_type,
                                "id": npc.type_id,
                                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                                "distance": dist,
                                "same_road": same_road_as_ego,
                                "same_dir": same_direction_as_ego,
                                "speed": speed
                            })

                        if display_bb:
                            for edge in edges:
                                p1 = get_image_point(verts[edge[0]], projection_matrix, world_2_camera)
                                p2 = get_image_point(verts[edge[1]], projection_matrix, world_2_camera)
                                if point_in_canvas(p1, image_h, image_w) and point_in_canvas(p2, image_h, image_w):
                                    cv2.line(
                                        img,
                                        (int(p1[0]), int(p1[1])),
                                        (int(p2[0]), int(p2[1])),
                                        (255, 0, 0, 255),
                                        1,
                                    )

    if display_bb:
        cv2.imshow("Scene with Bounding Boxes", img)
        cv2.waitKey(1)

    return scene_description

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

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
        
        correct_control, incorrect_control, ec = agent.run_step()
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
        scene_description = get_scene_description_and_bounding_boxes(
            world=world,
            vehicle=ego_vehicle,
            camera=rgb_cam.get_sensor(),
            image=rgb_cam.get_sensor_data(),
            image_h=512,
            image_w=1024,
            K=build_projection_matrix(1024, 512, 110.0),
            K_b=build_projection_matrix(1024, 512, 110.0, is_behind_camera=True)
        )
        label = generate_label(weather, correct_control.steer, correct_control.brake, correct_control.throttle, light, waypoint, ec, scene_description, collect_correct)

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
    parser.add_argument('--image_path', type=str, default='/vlm_data/images/', help='Filepath for images')
    args = parser.parse_args()

    main(args)