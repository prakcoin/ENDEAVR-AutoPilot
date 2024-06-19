import torch
import carla
import numpy as np
from model import AVModel
from utils import init_world, get_spawn_point, spawn_ego_vehicle, start_camera, process_image, update_spectator

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

model = AVModel()
model.load_state_dict(torch.load('best_model.pt',  map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

world, client = init_world('Town01', 'ClearNoon')
spawn_point = get_spawn_point(world)

ego_vehicle = spawn_ego_vehicle(world, spawn_point)
camera = start_camera(world, ego_vehicle, carla.Transform(carla.Location(2,0,1)), process_image, model)
spectator = world.get_spectator()

try:
    while True:
        world.tick()
        update_spectator(spectator, ego_vehicle)
except KeyboardInterrupt:
    pass
finally:
    ego_vehicle.destroy()
    camera.destroy()
    print("Simulation ended.")
