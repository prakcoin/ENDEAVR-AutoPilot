import os
import argparse
import torch
from utils.shared_utils import init_world, create_route, set_red_light_time, spawn_ego_vehicle, setup_traffic_manager, cleanup, update_spectator
from utils.evaluation_utils import start_camera, model_control, load_model

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

def main(args):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    model_path = os.path.join(parent_directory, 'ENDEAVR-AutoPilot', 'model', 'saved_models', 'av_model.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    world, client = init_world(args.town, args.weather)
    traffic_manager = setup_traffic_manager(client)
    set_red_light_time(world)
    spawn_point, route = create_route(world)

    ego_vehicle = spawn_ego_vehicle(world, spawn_point)
    camera = start_camera(world, ego_vehicle)
    traffic_manager.set_path(ego_vehicle, route)
    spectator = world.get_spectator()

    try:
        for _ in range(10):
            world.tick()
        while True:
            transform = ego_vehicle.get_transform()
            vehicle_location = transform.location
            world.tick()
            control = model_control(camera, model)
            print(f"Steering: {control.steer}, Throttle: {control.throttle}, Brake: {control.brake}")
            ego_vehicle.apply_control(control)
            update_spectator(spectator, ego_vehicle)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(ego_vehicle, camera)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('-t', '--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('-w', '--weather', type=str, default='ClearNoon', help='Weather condition to set')
    args = parser.parse_args()
    main(args)