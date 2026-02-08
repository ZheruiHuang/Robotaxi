from dispatch import Dispatch
import json
from tqdm import tqdm
import time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import argparse
import os.path as osp
import logging
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--private_car_route_file", type=str, required=True)
parser.add_argument("--align_files_dir", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--radius", type=float, default=200)
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

waitingRequestTime = 60
CUSTOMER_PER_TRIP = 4
RADIUS = args.radius
MAX_PERCENT_ROBOTAXI = 1
dispatch = Dispatch(None, CUSTOMER_PER_TRIP, RADIUS,
                    osp.join(args.align_files_dir, "align_node.csv"),
                    osp.join(args.align_files_dir, "align_edge.csv"))
dispatch.wander_steps = 5

request_lst = []
with open(args.private_car_route_file, "r") as f:
    n_requests = int(f.readline().strip())
    for _ in range(n_requests):
        timestamp = int(f.readline().strip().split()[0])
        n_routes = int(f.readline().strip())
        route = list(map(int, f.readline().strip().split()))
        request = {
            "time": timestamp,
            "origin": dispatch.get_coord_from_edge(route[0], from_node=True),
            "destination": dispatch.get_coord_from_edge(route[-1], from_node=False)
        }
        request_lst.append(request)
request_lst = sorted(request_lst, key=lambda x: x["time"])

def cluster_requests(request_lst):
    n_robotaxi = 0
    clustered_requests = []
    n_requests = len(request_lst)
    request_lst = deepcopy(request_lst)
    while request_lst:
        base_request = request_lst.pop(0)
        base_origin = base_request['origin']
        base_dest = base_request['destination']
        cluster = [base_request]
        if n_robotaxi+1 <= MAX_PERCENT_ROBOTAXI * (len(clustered_requests)+1):
            to_remove = []
            for request in request_lst:
                request_origin = request['origin']
                request_dest = request['destination']
                if request['time'] // waitingRequestTime == base_request['time'] // waitingRequestTime:
                    if dispatch._calculate_distance(base_origin[0], base_origin[1], request_origin[0], request_origin[1]) <= RADIUS and \
                    dispatch._calculate_distance(base_dest[0], base_dest[1], request_dest[0], request_dest[1]) <= RADIUS:
                        cluster.append(request)
                        to_remove.append(request)
                        if len(cluster) == CUSTOMER_PER_TRIP:
                            break
                else:
                    break
            for request in to_remove:
                request_lst.remove(request)
            if len(cluster) > 1:
                n_robotaxi += 1
            cluster.insert(0, "delay")
        clustered_requests.append(cluster)

    extra_waiting_time = 0
    for cluster in clustered_requests:
        if cluster[0] == "delay":
            for request in cluster[1:]:
                extra_waiting_time += ((request['time']//waitingRequestTime) + 1) * waitingRequestTime - request['time']
    logger.info(f"Avg extra waiting time: {extra_waiting_time / n_requests}")
    stats = [0 for _ in range(CUSTOMER_PER_TRIP+1)]
    for cluster in clustered_requests:
        l = len(cluster) if cluster[0] != "delay" else len(cluster) - 1
        stats[l] += 1
    logger.info(f"Stats: {stats}, >1 percentage: {sum(stats[2:]) / n_requests}")
    return clustered_requests

def process_request(request):
    if request[0] == "delay":
        request.pop(0)
        timestamp = ((request[-1]["time"] // waitingRequestTime) + 1) * waitingRequestTime
    else:
        timestamp = request[-1]["time"]
    origin = [r["origin"] for r in request]
    dest = [r["destination"] for r in request]
    route, distance, start_idcs, end_idcs = dispatch.apply_route(None, origin, dest)
    start_idcs = [idx - dispatch.wander_steps for idx in start_idcs]
    end_idcs = [idx - dispatch.wander_steps for idx in end_idcs]
    if len(route) <= dispatch.wander_steps + 1:
        return None
    return (timestamp, route[1:-dispatch.wander_steps], distance, start_idcs, end_idcs)

def main():
    start_time = time.time()
    logger.info(f"Start time: {datetime.now()}")
    # logger.info(f"Request file: {args.private_car_requests_file}")
    logger.info(f"Align files dir: {args.align_files_dir}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Waiting request time: {waitingRequestTime}")
    logger.info(f"Max percent robotaxi: {MAX_PERCENT_ROBOTAXI}")
    logger.info(f"Customer per trip: {CUSTOMER_PER_TRIP}")
    logger.info(f"Radius: {RADIUS}")

    routes = []
    cluster_requests_lst = cluster_requests(request_lst)
    print(len(request_lst), len(cluster_requests_lst))
    dispatch.wander_steps = 5

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_request, cluster_requests_lst), total=len(cluster_requests_lst)))

    for result in results:
        if result is not None:
            routes.append(result)

    # for cluster in tqdm(cluster_requests_lst):
    #     result = process_request(cluster)
    #     if result is not None:
    #         routes.append(result)

    routes = sorted(routes, key=lambda x: x[0])

    with open(args.output_file, "w") as f:
        f.write(str(len(routes)) + "\n")
        for timestamp, route, distance, start_idcs, end_idcs in routes:
            f.write(f"{timestamp} {timestamp+2} 10\n")
            f.write(str(len(route)) + "\n")
            f.write(" ".join(map(str, route)) + "\n")

    json_lst = []
    for timestamp, route, distance, start_idcs, end_idcs in routes:
        json_lst.append({
            "time": int(timestamp),
            "route": list(map(int, route)),
            "distance": float(distance),
            "start_idcs": list(map(int, start_idcs)),
            "end_idcs": list(map(int, end_idcs))
        })

    with open(f"{args.output_file}.json", "w") as f:
        json.dump(json_lst, f)

    logger.info(f"Output file: {args.output_file}.json")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
