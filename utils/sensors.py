import carla
import cv2
import numpy as np

class RGBCamera:
    def __init__(self, world, vehicle, size_x='320', size_y='240', fov='90', x_pos=1.5, y_pos=0.0, z_pos=2.4):
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_transform = carla.Transform(
            carla.Location(x=x_pos, y=y_pos, z=z_pos),
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
    def __init__(self, world, vehicle, size_x='320', size_y='240', fov='90', x_pos=1.5, y_pos=0.0, z_pos=2.4):
        cam_bp = world.get_blueprint_library().find('sensor.camera.depth')
        cam_transform = carla.Transform(
            carla.Location(x=x_pos, y=y_pos, z=z_pos),
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
    rgb_cam_main = RGBCamera(world, vehicle, size_x='320', size_y='240', fov='90', x_pos=1.5, y_pos=0, z_pos=2.4)
    depth_cam = DepthCamera(world, vehicle, size_x='320', size_y='240', fov='90', x_pos=1.5, y_pos=0, z_pos=2.4)
    return rgb_cam_main, depth_cam

def start_collision_sensor(world, vehicle):
    bp = world.get_blueprint_library().find('sensor.other.collision')
    transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    return sensor