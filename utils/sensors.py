import carla

class RGBCamera:
    def __init__(self, world, vehicle, size_x='288', size_y='200', fov='90'):
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))
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
    # narrow_rgb_cam = RGBCamera(world, vehicle, size_x='224', size_y='224', fov='60')
    main_rgb_cam = RGBCamera(world, vehicle, size_x='288', size_y='200', fov='90')
    #wide_rgb_cam = RGBCamera(world, vehicle, size_x='224', size_y='224', fov='150')
    return main_rgb_cam

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