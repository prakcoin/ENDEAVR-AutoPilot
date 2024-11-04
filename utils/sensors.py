import carla
import cv2
import numpy as np
import open3d as o3d

class RGBCamera:
    def __init__(self, world, vehicle, size_x='320', size_y='240', fov='90', y_offset=0.0):
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

class DepthCamera:
    def __init__(self, world, vehicle, size_x='320', size_y='240', fov='90', y_offset=0.0):
        cam_bp = world.get_blueprint_library().find('sensor.camera.depth')
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
    rgb_cam_main = RGBCamera(world, vehicle, size_x='320', size_y='240', fov='90', y_offset=0)
    rgb_cam_left = RGBCamera(world, vehicle, size_x='320', size_y='240', fov='90', y_offset=-baseline / 2)
    rgb_cam_right = RGBCamera(world, vehicle, size_x='320', size_y='240', fov='90', y_offset=baseline / 2)
    return rgb_cam_main, rgb_cam_left, rgb_cam_right

def k_matrix():
    image_w = 320
    image_h = 240
    fov = 90.0
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0
    return K

def compute_left_disparity(img_left, img_right):
    block_size = 11
    win_size = 1
    min_disparity = 0
    n_disp_factor = 2
    num_disparities = 16 * n_disp_factor - min_disparity

    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * win_size**2,
        P2=32 * 3 * win_size**2,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=2,
        disp12MaxDiff=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp_left = left_matcher_SGBM.compute(img_left_gray, img_right_gray).astype(np.float32) / 16
    return disp_left, left_matcher_SGBM

def compute_right_disparity(img_left, img_right):
    block_size = 11
    win_size = 1
    min_disparity = 0
    n_disp_factor = 2
    num_disparities = 16 * n_disp_factor - min_disparity

    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    right_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * win_size**2,
        P2=32 * 3 * win_size**2,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=2,
        disp12MaxDiff=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp_right = right_matcher_SGBM.compute(img_right_gray, img_left_gray).astype(np.float32) / 16
    return disp_right

def apply_wls_filter(disp_left, disp_right, left_matcher, img_left):
    lmbda = 8000
    sigma = 1.5

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    filtered_disp = wls_filter.filter(disp_left, img_left, disparity_map_right=disp_right)
    return filtered_disp

def calculate_depth(lf, rf):
    disp_left, left_matcher = compute_left_disparity(lf, rf)
    disp_right = compute_right_disparity(lf, rf)
    
    disp_left[disp_left <= 0.1] = 0.1
    disp_right[disp_right <= 0.1] = 0.1

    disp_left_filtered = apply_wls_filter(disp_left, disp_right, left_matcher, lf)
    
    f = k_matrix()[0, 0]
    b = 0.4

    depth_map = np.ones(disp_left_filtered.shape, np.single)
    depth_map[:] = f * b / disp_left_filtered[:]

    max_depth = 50.0
    depth_map = np.clip(depth_map, 0, max_depth)

    # depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR) if len(main.shape) == 2 else main
    # gt_depth = cv2.cvtColor(gt_depth, cv2.COLOR_GRAY2BGR) if len(gt_depth.shape) == 2 else gt_depth
    # depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

    # grid_image = np.hstack([main, depth_map, gt_depth])

    # cv2.imshow("Comparison Grid (Main | Computed Depth | GT Depth)", grid_image)
    # cv2.waitKey(1)

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