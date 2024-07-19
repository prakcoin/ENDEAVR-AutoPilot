import numpy as np

class DistanceTracker:
    def __init__(self):
        self.prev_location = None
        self.total_distance = 0.0

    def update(self, vehicle):
        location = vehicle.get_transform().location
        if self.prev_location is not None:
            self.total_distance += np.sqrt((location.x - self.prev_location.x) ** 2 +
                                             (location.y - self.prev_location.y) ** 2)
        self.prev_location = location

    def get_total_distance(self):
        return self.total_distance