import csv
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
args = parser.parse_args()

with open(osp.join(args.data_dir, "roadnet.txt"), "r") as f:
    node_n = int(f.readline().strip())
    node_lst = []
    for i in range(node_n):
        lat, lon, ID = f.readline().strip().split()[:3]
        node_lst.append([int(ID), float(lat), float(lon)])
    edge_n = int(f.readline().strip())
    edge_lst = []
    for i in range(edge_n):
        edge = f.readline().strip().split()
        from_id, to_id, ID1, ID2 = edge[0], edge[1], edge[-2], edge[-1]
        edge_lst.append([int(ID1), int(from_id), int(to_id)])
        edge_lst.append([int(ID2), int(to_id), int(from_id)])
        f.readline()
        f.readline()

with open(osp.join(args.data_dir, "align_node.csv"), "w") as f:
    writer = csv.writer(f)
    for node in node_lst:
        writer.writerow(node)

with open(osp.join(args.data_dir, "align_edge.csv"), "w") as f:
    writer = csv.writer(f)
    for edge in edge_lst:
        writer.writerow(edge)