import carla

class RGBCamera:
    def __init__(self, world, vehicle, location, size_x='224', size_y='224'):
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_transform = carla.Transform(carla.Location(2, 0, 1))
        cam_bp.set_attribute('image_size_x', size_x)
        cam_bp.set_attribute('image_size_y', size_y)

        self._sensor = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        self._data = None
        self._sensor.listen(lambda data: self._callback(data))

    def _callback(self, data):
        self._data = data
    
    def get_sensor_data(self):
        return self._data

    def get_sensor(self):
        return self._sensor
    
def start_cameras(world, vehicle):
    narrow_camera_location = carla.Location(2, 0, 1)
    main_camera_location = carla.Location(2, 0, 1)
    wide_camera_location = carla.Location(2, 0, 1)
    narrow_rgb_cam = RGBCamera(world, vehicle, narrow_camera_location, size_x='224', size_y='224', fov=60)
    main_rgb_cam = RGBCamera(world, vehicle, main_camera_location, size_x='224', size_y='224', fov=90)
    narrow_rgb_cam = RGBCamera(world, vehicle, wide_camera_location, size_x='224', size_y='224', fov=120)
    return [narrow_rgb_cam, main_rgb_cam, narrow_rgb_cam]

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