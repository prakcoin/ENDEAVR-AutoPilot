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

def generate_scene_description(scene_description, ego_speed):
    descriptions = []
    vehicle_count, ped_count = 0, 0

    for obj in scene_description:
        distance = obj["distance"]
        if distance < 10:
            proximity_str = "very close to the ego vehicle"
        elif distance < 20:
            proximity_str = "close to the ego vehicle"
        elif distance < 35:
            proximity_str = "at a moderate distance from the ego vehicle"
        else:
            proximity_str = "far away from the ego vehicle"


        if obj["type"] == "vehicle":
            if (
                obj['position'][0] < -1.5
                or (obj['base_type'] != 'bicycle' and obj['num_points'] < 15)
                or (obj['num_points'] < 10)
            ):
                continue

            if not is_vehicle_visible_in_image(obj):
                continue

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
            
            if ego_speed > 0.2 and motion_status == "stopped" and rough_pos_str == "directly in front of the ego vehicle" and distance < 20:
                caution_str = f" The {vehicle_type.lower()} ahead is stopped, potentially causing a collision if evasive action isn't taken."
            if ego_speed > obj['speed'] and motion_status == "moving slowly" and rough_pos_str == "directly in front of the ego vehicle" and distance < 20:
                caution_str = f" The {vehicle_type.lower()} ahead is slowing down, potentially causing a collision if evasive action isn't taken."
            else:
                caution_str = ""

            desc = f"A {obj['color'].lower()} {vehicle_type.lower()}" if obj['color'] else f"A {vehicle_type}"
            desc += f" is {motion_status}, {turning_status}, {orientation_str}, located {rough_pos_str}, and is {proximity_str}.{caution_str}"
            vehicle_count += 1
            descriptions.append(desc)

        elif obj["type"] == "pedestrian":
            if (
                obj['num_points'] < 5
                or obj['position'][0] < 1
                or obj['position'][0] > 50
            ):
                continue

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

            if ego_speed > 0.2 and motion_status == "walking" and rough_pos_str == "directly in front of the ego vehicle" and distance < 20:
                caution_str = " The pedestrian is crossing in front of the ego vehicle, potentially causing a collision if evasive action isn't taken."
            else:
                caution_str = ""

            desc = f"A pedestrian is {motion_status}, located {rough_pos_str}, and is {proximity_str}.{caution_str}"
            ped_count += 1
            descriptions.append(desc)

    vehicle_count_str = f"are {vehicle_count} vehicles"
    if vehicle_count == 1:
        vehicle_count_str = f"is {vehicle_count} vehicle"
    
    pedestrian_count_str = f"{ped_count} pedestrians"
    if ped_count == 1:
        pedestrian_count_str = f"{ped_count} pedestrian"
    
    descriptions.insert(0, f"There {vehicle_count_str} and {pedestrian_count_str} nearby.")

    return " ".join(descriptions)

def control_signal_desc(steer, brake, throttle):
    if steer < -0.1:
        turning_status = "turning left"
    elif steer < -0.03:
        turning_status = "slightly turning left"
    elif steer > 0.1:
        turning_status = "turning right"
    elif steer > 0.03:
        turning_status = "slightly turning right"
    else:
        turning_status = "going straight"

    if brake == 1.0:
        motion_status = "stopped"
    elif brake > 0.5:
        motion_status = "decelerating"
    elif brake > 0.1:
        motion_status = "lightly decelerating"
    elif brake > 0.0:
        motion_status = "barely decelerating"
    elif throttle == 1.0:
        motion_status = "accelerating at full throttle"
    elif throttle > 0.5:
        motion_status = "accelerating"
    elif throttle > 0.1:
        motion_status = "lightly accelerating"
    elif throttle > 0.0:
        motion_status = "barely accelerating"
    else:
        motion_status = "maintaining speed" 
    
    return f"{motion_status} while {turning_status}"

def generate_explanation(correct_steer, correct_brake, correct_throttle,  
                         ec, collect_correct, incorrect_steer=None, incorrect_brake=None, incorrect_throttle=None):
    explanation = "Based on the current scene and high-level command, "

    correct_signal_desc = control_signal_desc(correct_steer, correct_brake, correct_throttle)

    if collect_correct:
        return explanation + f"the ego vehicle should be {correct_signal_desc}, which aligns with the predicted control signals."
    else:
        incorrect_signal_desc = control_signal_desc(incorrect_steer, incorrect_brake, incorrect_throttle)
        if correct_signal_desc == incorrect_signal_desc:
            explanation += f"the ego vehicle should be {correct_signal_desc}, but "

            if ec == "plus_right_steer":
                explanation += "the model predicted excessive rightward steering, which could cause the vehicle to drift out of its lane or overturn."
            elif ec == "plus_left_steer":
                explanation += "the model predicted excessive leftward steering, which could cause the vehicle to drift out of its lane or overturn."
            elif ec == "plus_throttle":
                explanation += "the model predicted excessive acceleration, potentially making it difficult to stop in time for obstacles ahead."
            elif ec == "minus_throttle":
                explanation += "the model predicted insufficient acceleration, which might slow down traffic."
            elif ec == "plus_brake":
                explanation += "the model predicted excessive braking, which could disrupt the vehicle's normal movement and affect traffic flow."
            elif ec == "minus_brake":
                explanation += "the model predicted insufficient braking, which could increase the risk of a collision."

        else:
            explanation += f"the ego vehicle should be {correct_signal_desc}, but predicted control signals indicate that the ego vehicle will be {incorrect_signal_desc}. This means that "

            if ec == "plus_right_steer":
                explanation += "the model predicted excessive rightward steering, which could cause the vehicle to drift out of its lane or overturn."
            elif ec == "plus_left_steer":
                explanation += "the model predicted excessive leftward steering, which could cause the vehicle to drift out of its lane or overturn."
            elif ec == "plus_throttle":
                explanation += "the model predicted excessive acceleration, potentially making it difficult to stop in time for obstacles ahead."
            elif ec == "minus_throttle":
                explanation += "the model predicted insufficient acceleration, which might slow down traffic."
            elif ec == "plus_brake":
                explanation += "the model predicted excessive braking, which could disrupt the vehicle's normal movement and affect traffic flow."
            elif ec == "minus_brake":
                explanation += "the model predicted insufficient braking, which could increase the risk of a collision."
            elif ec == "swap_throttle":
                explanation += "the model predicted braking instead of acceleration, which could disrupt the vehicle's normal movement and affect traffic flow."
            elif ec == "swap_brake":
                explanation += "the model predicted acceleration instead of braking, which could increase the risk of a collision."

    return explanation

def generate_label(world, ego_vehicle, weather, hlc, speed, correct_steer, correct_brake, correct_throttle, waypoint, ec, scene_description, collect_correct, incorrect_steer=None, incorrect_brake=None, incorrect_throttle=None):
    weather_conditions = {
        "ClearNoon": "The weather is clear and sunny at noon.",
        "CloudyNoon": "The weather is cloudy at noon.",
        "WetNoon": "The ground is wet, but there is no rain at noon.",
        "WetCloudyNoon": "It is wet and cloudy at noon.",
        "MidRainyNoon": "There is moderate rain at noon.",
        "HardRainNoon": "It is heavily raining at noon.",
        "SoftRainNoon": "It is softly raining at noon.",        
        "ClearSunset": "The weather is clear at sunset.",
        "CloudySunset": "The weather is cloudy at sunset.",
        "WetSunset": "The ground is wet, but there is no rain at sunset.",
        "WetCloudySunset": "It is wet and cloudy at sunset.",
        "MidRainSunset": "There is moderate rain at sunset.",
        "HardRainSunset": "It is heavily raining at sunset.",
        "SoftRainSunset": "It is softly raining at sunset.",
        "ClearNight": "The weather is clear at night.",
        "CloudyNight": "The weather is cloudy at night.",
        "WetNight": "The ground is wet, but there is no rain at night.",
        "WetCloudyNight": "It is wet and cloudy at night.",
        "SoftRainNight": "It is softly raining at night.",
        "MidRainyNight": "There is moderate rain at night.",
        "HardRainNight": "It is heavily raining at night.",
        "DustStorm": "There is a dust storm."
    }
    road_option_dict = {
        "LaneFollow": "The high-level command is to follow the lane, ",
        "Left": "The high-level command is to turn left at the junction, so the ego vehicle should steer to the left, gradually increasing the negative steer value to smoothly follow the turn.",
        "Right": "The high-level command is to turn right at the junction, so the ego vehicle should steer to the right, gradually increasing the positive steer value to smoothly follow the turn.",
        "Straight": "The high-level command is to go straight at the junction, so the ego vehicle should maintain a near-zero steer value to stay on a straight path."
    }
    lang_hlc = road_option_dict[hlc]
    if hlc == "LaneFollow" and correct_steer < -0.03:
        lang_hlc += " since the road curves left, the ego vehicle should steer to the left, gradually increasing the negative steer value to smoothly follow the turn."
    elif hlc == "LaneFollow" and correct_steer > 0.03:
        lang_hlc += " since the road curves right, the ego vehicle should steer to the right, gradually increasing the positive steer value to smoothly follow the turn."
    else:
        lang_hlc += " so the ego vehicle should maintain a near-zero steer value to stay on a straight path."

    lang_scene = generate_scene_description(scene_description, speed)
    
    explanation = generate_explanation(correct_steer, correct_brake, correct_throttle, ec, collect_correct, incorrect_steer, incorrect_brake, incorrect_throttle)

    lang_weather = weather_conditions[weather]
    lang_light = light_affects_ego(world, ego_vehicle, speed)
    at_junction = waypoint.is_junction
    
    if at_junction:
        if scene_description:
            road_description = "The ego vehicle is at a junction and should be wary of any oncoming vehicles."
        else:
            road_description = "The ego vehicle is at a junction."
    else:
        lane_type = waypoint.lane_type.name.lower()
        left_lane_marking = waypoint.left_lane_marking.type.name.lower()
        
        road_description = (
            f"The ego vehicle isn't at a junction, and the road is a {lane_type} road with a {left_lane_marking} left lane marking."
        )

    label = (
        f"{lang_weather} {lang_light} {road_description} {lang_scene} {lang_hlc} {explanation} "
        f"Therefore, the appropriate control signals are:\n\n"
        f"- Steer: {correct_steer:.3f}\n"
        f"- Brake: {correct_brake:.3f}\n"
        f"- Throttle: {correct_throttle:.3f}"
    )
    return label

def get_relative_transform(ego_matrix, vehicle_matrix):
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    relative_pos = rot @ relative_pos

    return relative_pos

def normalize_angle_degree(x):
    x = x % 360.0
    if x > 180.0:
        x -= 360.0
    return x

def normalize_angle(x):
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x

def is_vehicle_visible_in_image(vehicle_obj):
    """
    Check if a vehicle is visible in the image.
    """
    # Project the 3D points of the vehicle onto the 2D image plane
    camera_matrix = build_projection_matrix(1024, 512, 110.0)
    projected_2d_points = project_center_corners(vehicle_obj, camera_matrix)
    min_x = 0
    max_x = 1024
    min_y = 0
    max_y = 384

    # Check if any projected point is visible
    vehicle_is_visible = False
    if projected_2d_points is None:
        return False

    for point_2d in projected_2d_points:
        if (point_2d[0] > min_x and point_2d[0] < max_x and
            point_2d[1] > min_y and point_2d[1] < max_y):
            vehicle_is_visible = True
            break

    return vehicle_is_visible

def lidar_to_ego_coordinate(lidar):
    """
    Converts the LiDAR points given by the simulator into the ego agents
    coordinate system
    :param config: GlobalConfig, used to read out lidar orientation and location
    :param lidar: the LiDAR point cloud as provided in the input of run_step
    :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
    coordinate system.
    """
    yaw = np.deg2rad(-90.0)
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

    translation = np.array([0.0, 0.0, 2.5])

    # The double transpose is a trick to compute all the points together.
    ego_lidar = (rotation_matrix @ lidar.T).T + translation

    return ego_lidar

def align_lidar(lidar, translation, yaw):
    """
    Translates and rotates a LiDAR into a new coordinate system.
    Rotation is inverse to translation and yaw
    :param lidar: numpy LiDAR point cloud (N,3)
    :param translation: translations in meters
    :param yaw: yaw angle in radians
    :return: numpy LiDAR point cloud in the new coordinate system.
    """
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    aligned_lidar = (rotation_matrix.T @ (lidar - translation).T).T
    return aligned_lidar

def get_points_in_bbox(vehicle_pos, vehicle_yaw, extent, lidar, type="vehicle"):
    """
    Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
    :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
    :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
    :param extent: List, Extent of the bounding box
    :param lidar: LiDAR point cloud
    :return: Returns the number of LiDAR hits within the bounding box of the
    vehicle
    """

    rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                                            [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

    # LiDAR in the with the vehicle as origin
    vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T
    x, y, z = extent[0], extent[1], extent[2]
    num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                                (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
    
    return num_points

def build_projection_matrix(w, h, fov):
    """
    Build a projection matrix based on image dimensions and field of view.
    
    Args:
        w (int): Image width
        h (int): Image height
        fov (float): Field of view in degrees
    
    Returns:
        np.ndarray: 3x3 projection matrix
    """
    focal = w / (2.0 * np.tan(np.radians(fov / 2)))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def project_center_corners(obj, K):
    """
    Project the center corners of an object onto the image plane.
    
    Args:
        obj (dict): Object dictionary containing position, extent, and yaw
        K (np.ndarray): Projection matrix
    
    Returns:
        np.ndarray: 2D array of projected corner points
    """
    pos = obj['position']
    if 'extent' not in obj:
        extent = [0.15,0.15,0.15]
    else:
        extent = obj['extent']
    if 'yaw' not in obj:
        yaw = 0
    else:
        yaw = -obj['yaw']
        
    # get bbox corners coordinates
    corners = np.array([[-extent[0], 0, 0.75],
                        [extent[0], 0, 0.75]])

    # rotate bbox
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]])
    corners = corners @ rotation_matrix.T

    # translate bbox
    corners = corners + np.array(pos)
    all_points_2d = []
    for corner in  corners:
        pos_3d = np.array([corner[1], -corner[2], corner[0]])
        rvec = np.zeros((3, 1), np.float32) 
        tvec = np.array([[0.0, 2.0, 1.5]], np.float32)
        # Define the distortion coefficients 
        dist_coeffs = np.zeros((5, 1), np.float32) 
        points_2d, _ = cv2.projectPoints(pos_3d, 
                            rvec, tvec, 
                            K, 
                            dist_coeffs)
        all_points_2d.append(points_2d[0][0])
        
    return np.array(all_points_2d)

def light_affects_ego(world, vehicle, speed):
    world_map = world.get_map()
    ego_wp = world_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
    ego_rotation = vehicle.get_transform().rotation
    ego_yaw = np.deg2rad(ego_rotation.yaw)
    
    if vehicle.is_at_traffic_light():
        light_dict = {
            carla.libcarla.TrafficLightState.Red: "The ego vehicle is at a red light and should remain stopped.",
            carla.libcarla.TrafficLightState.Green: "The ego vehicle is at a green light and should proceed.",
            carla.libcarla.TrafficLightState.Yellow: "The ego vehicle is at a yellow light and should proceed with caution."
        }
        traffic_light = vehicle.get_traffic_light()
        light_status = traffic_light.get_state()
        tl_state_vehicle = light_dict[light_status]
    else:
        nearby_tls = world.get_traffic_lights_from_waypoint(ego_wp, 50.0)
        tl_state_vehicle = 'None'
        if len(nearby_tls) == 0:
            tl_state_vehicle = 'None'
        else:
            for tl in nearby_tls:
                tl_wp = world_map.get_waypoint(tl.get_location(), project_to_road=True)
                tl_rotation = tl.get_transform().rotation
                tl_yaw = np.deg2rad(tl_rotation.yaw)
                relative_yaw = normalize_angle(tl_yaw - ego_yaw)

                orientation_relative_to_ego = relative_yaw * 180 / np.pi 
                
                if tl_wp.road_id == ego_wp.road_id and (45 < orientation_relative_to_ego or orientation_relative_to_ego < 135):
                    tl_state_vehicle = str(tl.state)
                    break
        light_dict = {
            "None": "The ego vehicle is not affected by a traffic light.",
            "Red": "The upcoming traffic light is red, so the ego vehicle should prepare to stop.",
            "Green": "The upcoming traffic light is green, so the ego vehicle should proceed.",
            "Yellow": "The upcoming traffic light is yellow, so the ego vehicle should slow down."
        }
        if (tl_state_vehicle == "Red"):
            if (speed < 0.2):
                return "The upcoming traffic light is red. The ego vehicle should remain stopped."
        tl_state_vehicle = light_dict[tl_state_vehicle]
    return tl_state_vehicle

def get_scene_description(world, ego_vehicle, lidar):
    actors = world.get_actors()
    vehicle_list = actors.filter('*vehicle*')
    ped_list = actors.filter('*walker*')

    world_map = world.get_map()
    ego_wp = world_map.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
    ego_matrix = np.array(ego_vehicle.get_transform().get_matrix())
    ego_rotation = ego_vehicle.get_transform().rotation
    ego_yaw = np.deg2rad(ego_rotation.yaw)
    ego_lane_direction = ego_wp.lane_id / abs(ego_wp.lane_id)

    scene_description = []
    for vehicle in vehicle_list:
        if vehicle.id != ego_vehicle.id:
            dist = vehicle.get_location().distance(ego_vehicle.get_location())

            if dist < 50.0:
                actor_type = "vehicle"
                base_type = vehicle.attributes['base_type']
                vehicle_wp = world_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
                vehicle_control = vehicle.get_control()
                vehicle_rotation = vehicle.get_transform().rotation
                vehicle_matrix = np.array(vehicle.get_transform().get_matrix())
                relative_pos = get_relative_transform(ego_matrix, vehicle_matrix)
                same_road_as_ego = False
                same_direction_as_ego = False
                direction = vehicle_wp.lane_id / abs(vehicle_wp.lane_id)
                speed = (3.6 * np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2 + vehicle.get_velocity().z**2))
                yaw = np.deg2rad(vehicle_rotation.yaw)
                relative_yaw = normalize_angle(yaw - ego_yaw)
                vehicle_extent = vehicle.bounding_box.extent
                vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]

                if not lidar is None:
                    num_in_bbox_points = get_points_in_bbox(relative_pos, relative_yaw, vehicle_extent_list, lidar)
                else:
                    num_in_bbox_points = -1
                if direction == ego_lane_direction:
                    same_direction_as_ego = True
                if vehicle_wp.road_id == ego_wp.road_id:
                    same_road_as_ego = True
                try:
                    rgb = tuple(map(int, vehicle.attributes['color'].split(',')))
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
                    "id": vehicle.type_id,
                    "distance": dist,
                    'extent': vehicle_extent_list,
                    'num_points': int(num_in_bbox_points),
                })
    for ped in ped_list:
        dist = ped.get_location().distance(ego_vehicle.get_location())
        if dist < 50.0:
            actor_type = "pedestrian"
            ped_wp = world_map.get_waypoint(ped.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
            ped_matrix = np.array(ped.get_transform().get_matrix())
            ped_relative_pos = get_relative_transform(ego_matrix, ped_matrix)
            same_road_as_ego = False
            same_direction_as_ego = False
            direction = ped_wp.lane_id / abs(ped_wp.lane_id)
            speed = (3.6 * np.sqrt(ped.get_velocity().x**2 + ped.get_velocity().y**2 + ped.get_velocity().z**2))
            ped_rotation = ped.get_transform().rotation
            ped_yaw = np.deg2rad(ped_rotation.yaw)
            ped_relative_yaw = normalize_angle(ped_yaw - ego_yaw)
            if direction == ego_lane_direction:
                same_direction_as_ego = True
            if ped_wp.road_id == ego_wp.road_id:
                same_road_as_ego = True
            ped_extent = ped.bounding_box.extent
            ped_extent.x = max(1.5, ped_extent.x)
            ped_extent.y = max(1.5, ped_extent.y)
            ped_extent_list = [ped_extent.x, ped_extent.y, ped_extent.z]

            if not lidar is None:
                num_in_bbox_points = get_points_in_bbox(ped_relative_pos, ped_relative_yaw, ped_extent_list, lidar, "ped")
            else:
                num_in_bbox_points = -1

            scene_description.append({
                "type": actor_type,
                "id": ped.type_id,
                "position": [ped_relative_pos[0], ped_relative_pos[1], ped_relative_pos[2]],
                "distance": dist,
                "same_road": same_road_as_ego,
                "same_dir": same_direction_as_ego,
                "speed": speed,
                'num_points': int(num_in_bbox_points),
                'extent': ped_extent_list
            })
    return scene_description

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

def run_episode(world, weather, ego_vehicle, agent, rgb_cam, lidar_sensor, end_point, collect_correct, episode, args):
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
        
        correct_control, incorrect_control, ec = agent.run_step()
        ego_vehicle.apply_control(correct_control)

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
        selected_control = correct_control if collect_correct else incorrect_control
        finetune_prompt = generate_prompt(hlc, speed_km_h, selected_control.steer, selected_control.brake, selected_control.throttle)
        scene_description = get_scene_description(world=world, ego_vehicle=ego_vehicle, lidar=lidar_360)
        if collect_correct:
            label = generate_label(world, ego_vehicle, weather, hlc, speed_km_h, correct_control.steer, correct_control.brake, correct_control.throttle, waypoint, ec, scene_description, collect_correct)
        else:
            label = generate_label(world, ego_vehicle, weather, hlc, speed_km_h, correct_control.steer, correct_control.brake, correct_control.throttle, waypoint, ec, scene_description, collect_correct, incorrect_control.steer, incorrect_control.brake, incorrect_control.throttle)

        correct_str = "correct" if collect_correct else "incorrect"
        image_filename = f"{args.town}_episode_{episode + 1}_{correct_str}_frame_{frame:06d}.jpg"
        images.append((image_filename, rgb_data))

        data.append({
            "image": f"{args.image_path}{image_filename}",
            "prompt": finetune_prompt,
            "label": label
        })

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
        "ClearNoon", "MidRainSunset",
        "CloudyNight", "WetSunset",
        "HardRainNoon", "SoftRainNight",
    ]
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
        lidar_sensor = start_lidar_sensor(world, ego_vehicle)
        sensors = [rgb_cam.get_sensor(), collision_sensor, lidar_sensor.get_sensor()]
        setup_vehicle_for_tm(traffic_manager, ego_vehicle)

        run_episode(world, weather_choice, ego_vehicle, agent, rgb_cam, lidar_sensor, end_point, collect_correct, episode, args)
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
    parser.add_argument('--episodes', type=int, default=6, help='Number of episodes to collect data for')
    parser.add_argument('--vehicles', type=int, default=80, help='Number of vehicles present')
    parser.add_argument('--pedestrians', type=int, default=40, help='Number of pedestrians present')
    parser.add_argument('--route_file', type=str, default='routes/Town01_VLM.txt', help='Filepath for route file')
    parser.add_argument('--image_path', type=str, default='/vlm data/images/', help='Filepath for images')
    args = parser.parse_args()

    main(args)