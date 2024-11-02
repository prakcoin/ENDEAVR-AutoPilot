import carla
import cv2
import numpy as np
import open3d as o3d

class RGBCamera:
    def __init__(self, world, vehicle, size_x='288', size_y='200', fov='90', y_offset=0.0):
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_transform = carla.Transform(
            carla.Location(x=1.5, y=y_offset, z=2.4),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        
        cam_bp.set_attribute('image_size_x', size_x)
        cam_bp.set_attribute('image_size_y', size_y)
        cam_bp.set_attribute('fov', fov)
        
        self._sensor = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        self._data = None
        self._sensor.listen(lambda data: self._callback(data))

    def _callback(self, data):
        self._data = data

    def get_sensor_data(self):
        return self._data

    def get_sensor(self):
        return self._sensor

def start_camera(world, vehicle):
    baseline = 0.4
    rgb_cam_main = RGBCamera(world, vehicle, size_x='400', size_y='300', fov='90', y_offset=0)
    rgb_cam_left = RGBCamera(world, vehicle, size_x='400', size_y='300', fov='90', y_offset=-baseline / 2)
    rgb_cam_right = RGBCamera(world, vehicle, size_x='400', size_y='300', fov='90', y_offset=baseline / 2)
    return rgb_cam_main, rgb_cam_left, rgb_cam_right

def k_matrix():
    image_w = 400
    image_h = 300
    fov = 90.0
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect ratio is 1
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0
    return K

def compute_left_disparity(img_left, img_right):
    block_size = 5
    min_disparity = 0
    n_disp_factor = 1
    num_disparities = 16 * n_disp_factor - min_disparity
    
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Stereo BM matcher
    # left_matcher_BM = cv2.StereoBM_create(numDisparities=num_disparities,
    #                                       blockSize=block_size)

    # disp_left = left_matcher_BM.compute(img_left, img_right).astype(np.float32)/16

    # Stereo SGBM matcher
    left_matcher_SGBM = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                                numDisparities=num_disparities,
                                                blockSize=block_size,
                                                P1=8 * 3 * block_size**2,
                                                P2=32 * 3 * block_size**2,
                                                # disp12MaxDiff=1,
                                                # speckleRange=2,
                                                # preFilterCap=63,
                                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32) / 16

    return disp_left

def calculate_depth(lf, rf, main):
    disp_left = compute_left_disparity(lf, rf)
    f = k_matrix()[0,0]
    b = 0.4
    disp_left[disp_left <= 0] = 0.1
    depth_map = np.ones(disp_left.shape, np.single)
    depth_map[:] = f * b / disp_left[:]

    depth_map = np.clip(depth_map, 0, 50)
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("main", main)
    cv2.imshow("depth", depth_map)
    cv2.waitKey(1)  # Ensure the image has time to render

    return depth_map


def depth_to_pseudo_lidar(depth_map):
    """
    Convert depth map to pseudo-LiDAR (3D point cloud).
    """
    h, w = depth_map.shape
    f = k_matrix()[0, 0]
    cx, cy = k_matrix()[0, 2], k_matrix()[1, 2]
    
    # Create a grid of (u, v) coordinates corresponding to each pixel in the depth map
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert depth map to 3D points
    z = depth_map.flatten()
    x = (u_coords.flatten() - cx) * z / f
    y = (v_coords.flatten() - cy) * z / f
    
    # Stack into a single 3D array of (x, y, z) points
    points_3d = np.stack((x, y, z), axis=1)
    
    # Filter out points with invalid depth values (e.g., due to min_disp threshold)
    points_3d = points_3d[points_3d[:, 2] > 0]
    #visualize_point_cloud(points_3d)
    return points_3d

def visualize_point_cloud(points_3d):
    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    # Optionally add colors (e.g., based on depth)
    max_depth = np.max(points_3d[:, 2])
    colors = np.zeros(points_3d.shape)
    colors[:, 0] = points_3d[:, 2] / max_depth  # Color based on depth (optional)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([point_cloud], window_name="Pseudo-LiDAR Point Cloud")

def start_collision_sensor(world, vehicle):
    bp = world.get_blueprint_library().find('sensor.other.collision')
    transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    return sensor

def start_lane_invasion_sensor(world, vehicle):
    bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
    transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    return sensor