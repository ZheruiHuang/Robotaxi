import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from robotaxi.dispatch import Dispatch
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--align_files_dir', type=str, required=True)
parser.add_argument('--private_car_route_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
args = parser.parse_args()


# city = "newyork_new"
# REQUESTS_PER_HOUR = 40000
CUSTOMER_PER_TRIP = 1
RADIUS = 1
dispatch = Dispatch(None, CUSTOMER_PER_TRIP, RADIUS,
                    osp.join(args.align_files_dir, "align_node.csv"), 
                    osp.join(args.align_files_dir, "align_edge.csv"))
dispatch.wander_steps = 5

routes = []
with open(args.private_car_route_file, "r") as f:
    n_routes = int(f.readline().strip())
    for _ in tqdm(range(n_routes)):
        timestep = int(f.readline().strip().split()[0])
        n_route = f.readline().strip()
        route = list(map(int, f.readline().strip().split()))
        distance = dispatch.calc_path_distance(route)
        start_idcs = [len(route)]
        end_idcs = [0]
        routes.append((timestep, route, distance, start_idcs, end_idcs))

json_lst = []
for timestamp, route, distance, start_idcs, end_idcs in routes:
    json_lst.append({
        "time": int(timestamp),
        "route": list(map(int, route)),
        "distance": float(distance),
        "start_idcs": list(map(int, start_idcs)),
        "end_idcs": list(map(int, end_idcs))
    })

with open(args.output_file, "w") as f:
    json.dump(json_lst, f)