import argparse
import queue
import os
from utils.shared_utils import init_world, setup_traffic_manager, setup_vehicle_for_tm, spawn_ego_vehicle, create_route, to_rgb, cleanup, update_spectator, read_routes
from utils.data_collection_utils import init_dirs_csv, queue_callback, start_camera
from utils.sensors import start_collision_sensor

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

has_collision = False
def collision_callback(data):
    global has_collision
    has_collision = True

def end_reached(ego_vehicle, end_point):
    vehicle_location = ego_vehicle.get_location()
    end_location = end_point.location

    if end_location is None:
        return False 

    distance = vehicle_location.distance(end_location)
    return distance < 1.0

def end_episode(ego_vehicle, end_point, frame, max_frames):
    done = False
    if end_reached(ego_vehicle, end_point):
        print("Target has been reached, episode ending")
        done = True
    elif frame >= max_frames:
        print("Maximum frames reached, episode ending")
        done = True
    elif has_collision:
        print.info("Collision detected, episode ending")
        done = True
    return done

def run_episode(world, ego_vehicle, rgb_sensor, end_point, frames, spectator, writer, csv_file, run_dir):
    global has_collision
    has_collision = False

    for _ in range(10):
        world.tick()

    frame = 0
    while True:
        try:
            if end_episode(ego_vehicle, end_point, frame, frames):
                break

            update_spectator(spectator, ego_vehicle)
            writer.writerow([ego_vehicle.get_control().steer, ego_vehicle.get_control().throttle, ego_vehicle.get_control().brake, frame])
            sensor_data = to_rgb(rgb_sensor.get_sensor_data())
            sensor_data.save_to_disk(os.path.join(run_dir, 'img', f'{frame}.png'))
            world.tick()
            frame += 1

        except KeyboardInterrupt:
            print("Simulation interrupted")
            cleanup(ego_vehicle, rgb_sensor, csv_file)

    print("Simulation complete")

def main(args):
    world, client = init_world(args.town, args.weather)
    traffic_manager = setup_traffic_manager(client)
    route_configs = read_routes()

    for episode in args.episodes:
        spawn_point, end_point, route = create_route(world, route_configs)
        run_dir, writer, csv_file  = init_dirs_csv(args.town, args.weather, args.episodes)

        ego_vehicle = spawn_ego_vehicle(world, spawn_point)
        rgb_sensor = start_camera(world, ego_vehicle)
        collision_sensor = start_collision_sensor(world, ego_vehicle)
        collision_sensor.listen(collision_callback)
        setup_vehicle_for_tm(traffic_manager, ego_vehicle, route)
        spectator = world.get_spectator()

        print(f'Episode: {episode + 1}')
        run_episode(world, ego_vehicle, rgb_sensor, end_point, args.frames, spectator, writer, csv_file, run_dir)
        cleanup(ego_vehicle, rgb_sensor, csv_file)

if __name__ == '__main__':
    towns = ['Town01', 'Town02', 'Town06']
    weather_conditions = ['ClearNoon', 'ClearSunset', 'ClearNight', 'CloudyNoon', 'CloudyNight', 'WetNoon', 
                        'WetSunset', 'WetNight', 'SoftRainNoon', 'SoftRainSunset', 'SoftRainNight', 
                        'MidRainyNoon', 'MidRainSunset', 'MidRainyNight', 'HardRainNoon']

    # Parsing town and weather arguments
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('-t', '--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('-w', '--weather', type=str, default='ClearNoon', help='Weather condition to set')
    parser.add_argument('-f', '--frames', type=int, default=5000, help='Number of frames to collect per episode')
    parser.add_argument('-e', '--episodes', type=int, default=5, help='Number of frames to collect per episode')
    args = parser.parse_args()

    main(args)