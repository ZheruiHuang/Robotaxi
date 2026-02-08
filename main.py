import os, sys
import time
import cbengine
from utils import Dataloader
from robotaxi.tracer import Tracer
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--trace_file', type=str, required=True)
parser.add_argument('--roadnet_file', type=str, required=True)
parser.add_argument('--cfg_file', type=str, required=True)
parser.add_argument('--logging', type=str, required=True)
args = parser.parse_args()

is_trace = True

dataloader = Dataloader(args.roadnet_file, None, None)

def setup_logger(log_file):
    logger = logging.getLogger('CBEngine')
    logger.setLevel(logging.INFO)
    
    # Create a file handler.
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Attach handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    logger = setup_logger(args.logging)
    
    running_step = 28800
    phase_time = 90
    engine = cbengine.Engine(args.cfg_file, 12)
    max_vehicle_num = 0
    if is_trace:
        tracer = Tracer(detail_file=args.trace_file, max_speed=20)

    logger.info('Simulation starts ...')
    start_time = time.time()
    for step in range(running_step):
        for intersection in dataloader.intersections.keys():
            engine.set_ttl_phase(intersection, (int(engine.get_current_time()) // phase_time) % 4 + 1)
        max_vehicle_num = max(max_vehicle_num, engine.get_vehicle_count())
        engine.next_step()
        if is_trace:
            tracer.trace(engine)
        logger.info(" time step: {}, number of vehicles: {}".format(step, engine.get_vehicle_count()))
        if step > 7200 and engine.get_vehicle_count() == 0:
            break

    end_time = time.time()
    logger.info(f'Simulation finishes. Runtime: {end_time - start_time}')

    # Redirect logger output to ./result/{city_name}.txt.
    city_name = args.roadnet_file.split('/')[-2]
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    result_file = f'./result/{city_name}.txt'
    logger = setup_logger(result_file)
    data_file = f'./result/{city_name}_{os.path.basename(args.trace_file)}'
    import json
    # Save tracer.speed_hist to JSON.
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(tracer.speed_hist, f, ensure_ascii=False, indent=4)

    logger.info(f"Data saved to {data_file}")   
    logger.info(f"Avg travel time: {engine.get_average_travel_time()}; Max vehicle number: {max_vehicle_num}")
    if is_trace:
        tracer.calc_and_print_info(logger)


if __name__ == '__main__':
    main()
