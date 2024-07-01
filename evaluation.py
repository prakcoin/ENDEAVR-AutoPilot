import argparse
import torch
from carla.evaluation.evaluation_utils import init_world, create_route, spawn_ego_vehicle, start_camera, model_control, update_spectator, setup_traffic_manager, load_model

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

def main(args):
    model_path = 'best_model.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    world, client = init_world(args.town, args.weather)
    traffic_manager = setup_traffic_manager(client)

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
            control = model_control(camera, model, ego_vehicle)
            print(f"Steering: {control.steer}, Throttle: {control.throttle}, Brake: {control.brake}")
            ego_vehicle.apply_control(control)
            update_spectator(spectator, ego_vehicle)
    except KeyboardInterrupt:
        pass
    finally:
        ego_vehicle.destroy()
        camera.destroy()
        print("Simulation ended.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('-t', '--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('-w', '--weather', type=str, default='ClearNoon', help='Weather condition to set')
    args = parser.parse_args()
    main(args)