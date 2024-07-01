import carla

class RGBCamera:
    def __init__(self, world, vehicle, size_x = '256', size_y = '256'):
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_transform = carla.Transform(carla.Location(2,0,1))
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
