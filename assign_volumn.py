import networkx as nx
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import math
import random
import heapq
import json
import os.path as osp


class volumeAssigner:
    def __init__(self, roadnet_path, city_dir, link_df):
        """
        Args:
            roadnet_path: str, path to roadnet.txt
            city_dir: str, directory for city-related files
            link_df: DataFrame, columns: ["linkid", "order", "lat", "long"]
        """
        random.seed(0)
        
        self.graph = nx.DiGraph()
        with open(roadnet_path, "r") as f:
            n_node = int(f.readline().strip())
            for i in range(n_node):
                lat, lon, _id, _ = f.readline().strip().split()
                self.graph.add_node(int(_id), lat=float(lat), lon=float(lon))
            n_edge = int(f.readline().strip())
            for i in range(n_edge):
                start, end, length, speed_limit, _, _, _id1, _id2 = f.readline().strip().split()
                self.graph.add_edge(int(start), int(end), length=float(length), _id=int(_id1))
                self.graph.add_edge(int(end), int(start), length=float(length), _id=int(_id2))
                f.readline()
                f.readline()

        # Compute strongly connected components.
        print(f"raw graph: {self.graph.number_of_nodes()}, {self.graph.number_of_edges()}")
        scc = list(nx.strongly_connected_components(self.graph))
        scc = sorted(scc, key=len, reverse=True)
        self.graph = self.graph.subgraph(scc[0]).copy()
        print(f"largest scc: {self.graph.number_of_nodes()}, {self.graph.number_of_edges()}")

        assert nx.is_strongly_connected(self.graph)

        self.get_od_path()
        self.linkId_to_edgeId = self.blid_links_to_edges(link_df=link_df, save=True, load=True, save_path=osp.join(city_dir, "linkId_to_edgeId.json"))

    
    def assign_volumes(self, record_df, interval, scale):
        """
        Args:
            record_df: DataFrame, columns: ["linkid", "flow"]
            interval: int, time interval during recording in seconds.
        
        Returns:
            od_volume: dict, key: (origin, destination), value: List
        """

        volumes = {}
        for i, row in record_df.iterrows():
            edge_id = self.linkId_to_edgeId.get(row["linkid"], None)
            if edge_id is not None:
                volumes[edge_id] = volumes.get(edge_id, 0) + math.ceil(row["flow"] * (interval / 3600) / scale)
        if not volumes:
            raise ValueError("No volumes to assign.")
        return self._assign_volumes(volumes)


    def get_od_path(self):
        self.od_path_nodes = dict(nx.shortest_path(self.graph, weight="length"))
        for origin in self.od_path_nodes:
            del self.od_path_nodes[origin][origin]
        self.od_path_edges = defaultdict(lambda : defaultdict(list))
        for origin in self.od_path_nodes:
            for destination in self.od_path_nodes[origin]:
                path = self.od_path_nodes[origin][destination]
                for i in range(len(path) - 1):
                    self.od_path_edges[origin][destination].append(self.graph.edges[path[i], path[i+1]]["_id"])
        self.od_path_edges = {origin: dict(self.od_path_edges[origin]) for origin in self.od_path_edges}


    def _assign_volumes(self, volumes: dict):
        """
        Estimate origin-destination pair with given volume, using the greedy algorithm.
        Each time, recover an OD pair with the largest volume, and remove one volume from corresponding edges.
        
        Args:
            volume: dict, key: edge_id, value: vloumn.

        Returns:
            od_volume: dict, key: (origin, destination), value: List
        """
        assert type(volumes) == dict

        od_volume = defaultdict(int)
        print(f"total volumes: {sum(volumes.values())}")
        pbar = tqdm(total=sum(volumes.values()), desc="assign volumes")

        od_entries = []
        for origin, dests in self.od_path_edges.items():
            for destination, path in dests.items():
                od_entries.append((origin, destination, path, len(path)))

        k = math.floor(sum(len(self.od_path_edges[o]) for o in self.od_path_edges) * 0.2)
        while volumes:
            candidates = []
            vol = volumes  # Local binding to reduce global lookups.
            for origin, destination, path, path_len in od_entries:
                # Score by edge presence; equivalent to sum(min(..., 1)).
                score = sum(1 for edge in path if edge in vol)
                if score > 0:
                    candidates.append((score, -path_len, path, (origin, destination)))

            if not candidates:
                break  # No assignable path remains.

            if k > 0 and len(candidates) > k:
                top_k_paths = heapq.nlargest(k, candidates)
            else:
                # If k == 0 or candidates are fewer than k, use all candidates.
                # Keep behavior consistent with old logic when k == 0.
                top_k_paths = candidates if k != 0 else []

            if not top_k_paths:
                break  # No assignable path remains.

            vol_change_flag = False
            while not vol_change_flag:
                # Randomly pick one path from the candidate pool.
                score, neg_len, max_vol_path, od = random.choice(top_k_paths)

                flow = min([volumes[edge] for edge in max_vol_path if edge in volumes])
                cnt = 0
                for edge in max_vol_path:
                    if edge in volumes:
                        volumes[edge] -= flow
                        cnt += 1
                        if volumes[edge] <= 0:
                            del volumes[edge]
                            vol_change_flag = True
                od_volume[od] += flow
                pbar.update(flow * cnt)

        pbar.close()
        return dict(od_volume)


    def blid_links_to_edges(self, link_df: pd.DataFrame, save=False, load=False, save_path=None):
        """
        Get volumes from DataFrame.
        
        Args:
            link_df: DataFrame, columns: ["link_id", "order", "lat", "long"]
            
        Returns:
            volumes: dict, key: edge_id, value: volume.
        """
        if load and save_path is not None and osp.exists(save_path):
            with open(save_path, "r") as f:
                linkId_to_edgeId = json.load(f)
            return {float(k): v for k, v in linkId_to_edgeId.items()}
        
        n_segments = 100
        edge_segments = {}
        # Split each edge into n_segments parts. Keep only one for reverse duplicates.
        for origin, destination in self.graph.edges:
            if (origin, destination) in edge_segments or (destination, origin) in edge_segments:
                continue
            segments = []
            lat, lon = self.graph.nodes[origin]["lat"], self.graph.nodes[origin]["lon"]
            dlat, dlon = (self.graph.nodes[destination]["lat"] - lat) / n_segments, (self.graph.nodes[destination]["lon"] - lon) / n_segments
            for i in range(n_segments + 1):
                segments.append((lat + i * dlat, lon + i * dlon))
            edge_segments[(origin, destination)] = segments
        
        # Group link_df by link_id.
        link_df["lat"], link_df["long"] = link_df["lat"].astype(float), link_df["long"].astype(float)
        link_df["order"] = link_df["order"].astype(int)
        link_groups = link_df.groupby("linkid")
        linkId_to_edgeId = {}
        for link_id, group in tqdm(link_groups, desc="bind links to edges"):
            min_loss, od = float("inf"), None
            for (origin, destination), segments in edge_segments.items():
                loss = self.get_loss(segments, group)
                if loss < min_loss:
                    min_loss = loss
                    od = (origin, destination)

            # Determine whether the edge direction should be reversed from order.
            d1 = self.graph.nodes[od[1]]["lat"] - self.graph.nodes[od[0]]["lat"]
            group = group.sort_values(by="order")
            d2 = group.iloc[-1]["lat"] - group.iloc[0]["lat"]
            if d1 * d2 < 0:
                od = (od[1], od[0])
            linkId_to_edgeId[link_id] = self.graph.edges[od]["_id"]

        if save and save_path is not None:
            with open(save_path, "w") as f:
                json.dump(linkId_to_edgeId, f)

        return linkId_to_edgeId

    
    def get_loss(self, segments: list, group: pd.DataFrame):
        loss = 0
        for i, row in group.iterrows():
            lat, lon = row["lat"], row["long"]
            loss += min([(lat-seg[0])**2 + (lon-seg[1])**2 for seg in segments])
        return loss
    
    def convert_ODVolume_to_requests(self, od_volume, time_interval, time_complete="random"):
        """
        Convert ODVolume to requests.
        
        Args:
            od_volume: dict, key: (origin, destination), value: volume.
            time: str, time of the request. Support "random".
        
        Returns:
            od_requests: List[dict], requests.
        """
        od_requests = []
        for (origin, destination), volume in od_volume.items():
            for _ in range(volume):
                oringin_lat, origin_lon = self.graph.nodes[origin]["lat"], self.graph.nodes[origin]["lon"]
                dest_lat, dest_lon = self.graph.nodes[destination]["lat"], self.graph.nodes[destination]["lon"]
                if time_complete == "random":
                    timestamp = random.randint(time_interval[0], time_interval[1])
                else:
                    raise ValueError("time should be 'random' or 'peak'.")
                od_requests.append({
                    "time": timestamp,
                    "origin": [oringin_lat, origin_lon],
                    "destination": [dest_lat, dest_lon],
                })
        return od_requests
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()

    DATA_ROOT = args.root_dir
    city = args.city
    flow_scale = args.scale

    roadnet_path = osp.join(DATA_ROOT, city, "roadnet.txt")
    link_df = pd.read_csv(osp.join(DATA_ROOT, city, "links.csv"))
    city_dir = osp.join(DATA_ROOT, city)
    volume_assigner = volumeAssigner(roadnet_path, city_dir, link_df)

    record_df = pd.read_csv(osp.join(DATA_ROOT, city, "utd19.csv"))
    record_df = record_df[record_df["error"] != 1]
    detector_df = pd.read_csv(osp.join(DATA_ROOT, city, "detectors.csv"))
    record_df = pd.merge(record_df, detector_df, on="detid")
    if not (record_df["city"] == city).all() and (record_df["citycode"] == city).all():
        print("City and citycode do not match the expected values.")
        record_df = record_df[(record_df["city"] == city) & (record_df["citycode"] == city)]
    record_df = record_df[["day", "interval", "flow", "linkid", "detid"]]
    print(record_df.shape)

    record_df = record_df[record_df["day"] == record_df["day"].iloc[0]]
    time_start, time_end, time_interval = 28800, 36000, 1200
    assert (time_end - time_start) % time_interval == 0
    record_df = record_df[(time_start <= record_df["interval"]) & (record_df["interval"] < time_end)]

    od_requests = []
    cur_time_start, cur_time_end = time_start, time_start + time_interval
    while cur_time_end <= time_end:
        cur_record_df = record_df[(record_df["interval"] >= cur_time_start) & (record_df["interval"] < cur_time_end)]
        cur_record_df = cur_record_df.groupby(["detid", "linkid"]).agg({"flow": "mean"}).reset_index()
        print(cur_record_df.shape)
        od_pairs = volume_assigner.assign_volumes(cur_record_df, interval=time_interval, scale=flow_scale)
        od_requests.extend(volume_assigner.convert_ODVolume_to_requests(od_pairs, time_interval=[cur_time_start, cur_time_end-1], time_complete="random"))
        cur_time_start += time_interval
        cur_time_end += time_interval
    
    print(f"Total requests: {len(od_requests)}")
    with open(osp.join(DATA_ROOT, city, "od_requests.json"), "w") as f:
        json.dump(od_requests, f)
    
