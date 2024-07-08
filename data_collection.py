import argparse
import queue
import os
from utils.shared_utils import init_world, setup_traffic_manager, setup_vehicle_for_tm, set_red_light_time, spawn_ego_vehicle, create_route, cleanup, update_spectator
from utils.data_collection_utils import init_dirs_csv, queue_callback, start_camera, end_collection

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
    setup_vehicle_for_tm(traffic_manager, ego_vehicle, route)

    spectator = world.get_spectator()

    for _ in range(10):
        world.tick()

    data_vals = {"steering": 0, "running": 0}
    running = True
    while running:
        world.tick()
        update_spectator(spectator, ego_vehicle)
        if not image_queue.empty() and not control_queue.empty():
            image = image_queue.get()
            control = control_queue.get()
            
            steering_needed, running_needed = end_collection(control, data_vals, args.steer_frames, args.running_frames)
            if steering_needed and (control[0] > 0.05 or control[0] < -0.05):
                data_vals["steering"] += 1
                writer.writerow([control[0], control[1], control[2], image.frame])
                image.save_to_disk(os.path.join(run_dir, 'img', f'{image.frame}.png'))

            if running_needed and control[1] > 0.0 and (-0.05 < control[0] < 0.05):
                data_vals["running"] += 1
                writer.writerow([control[0], control[1], control[2], image.frame])
                image.save_to_disk(os.path.join(run_dir, 'img', f'{image.frame}.png'))

            running = steering_needed or running_needed

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
    parser.add_argument('-sf', '--steer_frames', type=int, default=15000, help='Number of steering frames to collect per vehicle')
    parser.add_argument('-rf', '--running_frames', type=int, default=35000, help='Number of running frames to collect per vehicle')
    args = parser.parse_args()

    main(args)