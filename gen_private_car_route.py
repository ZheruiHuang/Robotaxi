from dispatch import Dispatch
import json
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor
import argparse
import os.path as osp
import logging
from datetime import datetime
import time

parser = argparse.ArgumentParser()
parser.add_argument("--request_file", type=str, required=True)
parser.add_argument("--align_files_dir", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()

log_file = f"{args.output_file}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

with open(args.request_file, "r") as f:
    request_lst = json.load(f)
selected_requests = []
for request in request_lst:
    if 28800 <= request["time"] < 36000:
        request["time"] -= 28800
        selected_requests.append(request)
request_lst = selected_requests
raw_request_8oclock = [request for request in request_lst if ( 0 <= request["time"] < 3600 )]
raw_request_9oclock = [request for request in request_lst if ( 3600 <= request["time"] < 7200 )]
logger.info(f"Number of requests in 8 o'clock: {len(raw_request_8oclock)}")
logger.info(f"Number of requests in 9 o'clock: {len(raw_request_9oclock)}")

random.seed(0)

request_lst = sorted(request_lst, key=lambda x: x["time"])

CUSTOMER_PER_TRIP = 1
RADIUS = 1
dispatch = Dispatch(None, CUSTOMER_PER_TRIP, RADIUS, 
                    osp.join(args.align_files_dir, "align_node.csv"),
                    osp.join(args.align_files_dir, "align_edge.csv"))
dispatch.wander_steps = 5

def process_request(request):
    timestamp = request["time"]
    origin, dest = [request["origin"]], [request["destination"]]
    try:
        route, _, _, _ = dispatch.apply_route(None, origin, dest)
    except:
        return None
    if len(route) <= dispatch.wander_steps + 1:
        return None
    return (timestamp, route[1:-dispatch.wander_steps])

def main():
    start_time = time.time()
    logger.info(f"Start time: {datetime.now()}")

    routes = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_request, request_lst), total=len(request_lst)))
    
    for result in results:
        if result is not None:
            routes.append(result)

    routes = sorted(routes, key=lambda x: x[0])

    with open(args.output_file, "w") as f:
        f.write(str(len(routes)) + "\n")
        for timestamp, route in routes:
            f.write(f"{timestamp} {timestamp+2} 10\n")
            f.write(str(len(route)) + "\n")
            f.write(" ".join(map(str, route)) + "\n")

    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()