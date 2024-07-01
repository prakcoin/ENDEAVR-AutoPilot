import argparse
import queue
import os
from utils.shared_utils import init_world, setup_traffic_manager, set_red_light_time, spawn_ego_vehicle, create_route, cleanup, update_spectator
from utils.data_collection_utils import init_dirs_csv, queue_callback, start_camera

# Windows: CarlaUE4.exe -carla-server-timeout=10000ms
# Linux: ./CarlaUE4.sh -carla-server-timeout=10000ms -RenderOffScreen

def main(args):
    world, client = init_world(args.town, args.weather)
    traffic_manager = setup_traffic_manager(client)
    set_red_light_time(world)
    spawn_point, route = create_route(world)

    image_queue = queue.Queue()
    control_queue = queue.Queue()

    run_dir, writer, csv_file  = init_dirs_csv(args.town, args.weather)

    ego_vehicle = spawn_ego_vehicle(world, spawn_point)
    camera = start_camera(world, ego_vehicle, callback=lambda image: queue_callback(image, image_queue, control_queue, ego_vehicle))
    traffic_manager.set_path(ego_vehicle, route)
    ego_vehicle.set_autopilot(True)
    spectator = world.get_spectator()

    for _ in range(10):
        world.tick()

    first_frame = None
    running = True
    while running:
        world.tick()
        update_spectator(spectator, ego_vehicle)
        if not image_queue.empty() and not control_queue.empty():
            image = image_queue.get()
            control = control_queue.get()
            
            # Record first frame encountered
            if first_frame is None:
                first_frame = image.frame
            
            writer.writerow([control[0], control[1], control[2], image.frame])
            image.save_to_disk(os.path.join(run_dir, 'img', f'{image.frame}.png'))

            if image.frame >= (first_frame + args.frames):
                running = False

    cleanup(ego_vehicle, camera, csv_file)

if __name__ == '__main__':
    towns = ['Town01', 'Town02', 'Town06']
    weather_conditions = ['ClearNoon', 'ClearSunset', 'ClearNight', 'CloudyNoon', 'CloudyNight', 'WetNoon', 
                        'WetSunset', 'WetNight', 'SoftRainNoon', 'SoftRainSunset', 'SoftRainNight', 
                        'MidRainyNoon', 'MidRainSunset', 'MidRainyNight', 'HardRainNoon']

    # Parsing town and weather arguments
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('-t', '--town', type=str, default='Town01', help='CARLA town to use')
    parser.add_argument('-w', '--weather', type=str, default='ClearNoon', help='Weather condition to set')
    parser.add_argument('-n', '--frames', type=int, default=7200, help='Number of frames to collect per vehicle')
    args = parser.parse_args()

    main(args)