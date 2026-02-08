import math
import pandas as pd
import heapq
import random

class Dispatch:
    def __init__(self, num_taxi, num_people_a_car, people_range, node_csv, edge_csv):
        self.free_vehicles = []
        self.busy_vehicles = []
        self.vehicle_status = {}
        self.attr_lst = ["loc", "dest", "status"]
        self.num_people_a_car = num_people_a_car
        self.people_range = people_range
        self.node_csv = node_csv
        self.edge_csv = edge_csv
        self.wander_steps = 1000
        self.read_graph()

    def add_vehicles(self, vehicles):
        # vehicles: list of tuples, each tuple contains the vehicle's ID and location
        for vehicle in vehicles:
            self.free_vehicles.append(vehicle[0])
            self.vehicle_status[vehicle[0]] = {"loc": vehicle[1], "dest": None}

    def aggregate_group(self, loc_lst, dest_lst):
        if len(loc_lst) != len(dest_lst):
            raise ValueError("loc_lst and dest_lst must have the same length")

        allocated_groups = []
        visited = [False] * len(loc_lst)

        for i in range(len(loc_lst)):
            if visited[i]:
                continue

            group = [i]
            visited[i] = True

            for j in range(i + 1, len(loc_lst)):
                if visited[j]:
                    continue

                if len(group) == self.num_people_a_car:
                    break

                if self._within_range(loc_lst[i], loc_lst[j]) and self._within_range(dest_lst[i], dest_lst[j]):
                    group.append(j)
                    visited[j] = True

            allocated_groups.append(group)

        return [[loc_lst[i] for i in group] for group in allocated_groups], \
            [[dest_lst[i] for i in group] for group in allocated_groups], \
            allocated_groups

    def allocate_taxi(self, loc_lst, dest_lst):
        loc = loc_lst[0]
        distance = float('inf')
        nearest_taxi = None
        for taxi in self.free_vehicles:
            taxi_loc = self.vehicle_status[taxi]["loc"]
            taxi_distance = self._calculate_distance(loc[0], loc[1], taxi_loc[0], taxi_loc[1])
            if taxi_distance < distance:
                distance = taxi_distance
                nearest_taxi = taxi
        if nearest_taxi is None:
            return None
        self.free_vehicles.remove(nearest_taxi)
        self.busy_vehicles.append(nearest_taxi)
        return nearest_taxi

    def get_out_degree(self, node_id):
        return self.edge_df[self.edge_df['from_id'] == node_id].shape[0]

    def find_nearest_node(self, loc, force_no_isolate=False):
        lat, lon = loc
        min_distance = float('inf')
        nearest_node = None
        for _, row in self.node_df.iterrows():
            node_id = row['node_id']
            if force_no_isolate and self.get_out_degree(node_id) <= 1:
                continue
            node_lat = row['lat']
            node_lon = row['lon']
            distance = self._calculate_distance(lat, lon, node_lat, node_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        return nearest_node

    def _within_range(self, point1, point2):
        lon1, lat1 = point1
        lon2, lat2 = point2
        distance = self._calculate_distance(lat1, lon1, lat2, lon2)
        return distance <= self.people_range

    def check_graph(self):
        # Drop edges whose endpoints are missing in node_df.
        self.edge_df = self.edge_df[self.edge_df['from_id'].isin(self.node_df['node_id'])]
        self.edge_df = self.edge_df[self.edge_df['to_id'].isin(self.node_df['node_id'])]
        # Drop self-loop edges where from_id equals to_id.
        # self.edge_df = self.edge_df[self.edge_df['from_id'] != self.edge_df['to_id']]
        # Drop nodes that are not referenced by any edge.
        self.node_df = self.node_df[self.node_df['node_id'].isin(self.edge_df['from_id']) | self.node_df['node_id'].isin(self.edge_df['to_id'])]

    def _set_isolate_node_unreachable(self):
        # Mark nodes with out-degree <= 1 as unreachable (avoid U-turns).
        last_removed_edges = set()
        while True:
            out_degree = self.edge_df['from_id'].value_counts()
            isolate_nodes = out_degree[out_degree <= 1].index
            if len(isolate_nodes) == 0 or set(isolate_nodes) == last_removed_edges:
                break
            last_removed_edges = set(isolate_nodes)
            for node in isolate_nodes:
                self.edge_df = self.edge_df[self.edge_df['to_id'] != node]
            print(f"Remove nodes with out degree <= 1: {len(isolate_nodes)}")

    def read_graph(self):
        self.node_df = pd.read_csv(self.node_csv, header=None)  # node_df: node_id, lat, lon
        self.edge_df = pd.read_csv(self.edge_csv, header=None)  # edge_df: edge_id, from_id, to_id
        # Assign column names for node_df and edge_df.
        self.node_df.columns = ['node_id', 'lat', 'lon']
        self.edge_df.columns = ['edge_id', 'from_id', 'to_id']

        # to_node = self.edge_df.loc[self.edge_df['edge_id'] == 866, 'to_id'].values[0]
        # info = self.edge_df[self.edge_df['from_id'] == to_node]
        # print(info)
        # exit()

        self.check_graph()
        self._set_isolate_node_unreachable()
        
        # Build the adjacency list.
        self.graph = {}
        for _, row in self.edge_df.iterrows():
            edge_id = row['edge_id']
            start_node = row['from_id']
            end_node = row['to_id']
            start_lat = self.node_df.loc[self.node_df['node_id'] == start_node, 'lat'].values[0]
            start_lon = self.node_df.loc[self.node_df['node_id'] == start_node, 'lon'].values[0]
            end_lat = self.node_df.loc[self.node_df['node_id'] == end_node, 'lat'].values[0]
            end_lon = self.node_df.loc[self.node_df['node_id'] == end_node, 'lon'].values[0]
            weight = self._calculate_distance(start_lat, start_lon, end_lat, end_lon)

            if start_node not in self.graph:
                self.graph[start_node] = []
            self.graph[start_node].append((end_node, weight, edge_id))

        # start_edge = 3201.0
        # loc_lst = [[41.179626, -8.60193], [41.179626, -8.60193], [41.179626, -8.60193], [41.179626, -8.60193]]
        # dest_lst = [[41.17695226779012, -8.603917977993042], [41.17515051252325, -8.60340090731642], [41.17516665458241, -8.607887310930412], [41.17323448539205, -8.603864036139397]]
        # path = self.apply_route(start_edge, loc_lst, dest_lst)
        # print(self.edge_df.loc[self.edge_df['edge_id'] == 308].values)
        # print(path[:10])
        # print(self._check_connectivity(path))
        # exit()

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        # Compute distance between two points with the Haversine formula.
        R = 6371  # Earth radius in kilometers.
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c  # Distance in kilometers.
        distance *= 1000  # Convert distance to meters.
        return distance

    def _get_reverse_edge(self, edge_id):
        from_node = self.edge_df.loc[self.edge_df['edge_id'] == edge_id, 'from_id'].values[0]
        to_node = self.edge_df.loc[self.edge_df['edge_id'] == edge_id, 'to_id'].values[0]
        reverse_edge = self.edge_df[(self.edge_df['from_id'] == to_node) & (self.edge_df['to_id'] == from_node)]
        if reverse_edge.shape[0] == 0:
            return None
        return reverse_edge['edge_id'].values[0]

    # Shortest-path algorithm.
    def shortest_path(self, start_node, end_node, init_prev_node):
        if end_node == start_node:
            return [], 0
        # node and edge_id
        distances = {edge_id: float("inf") for edge_id in self.edge_df['edge_id'].tolist()}
        init_prev_edge = self.edge_df.loc[(self.edge_df['from_id'] == init_prev_node) & (self.edge_df['to_id'] == start_node), 'edge_id'].values[0]
        priority_queue = [(0, init_prev_edge, start_node)]
        heapq.heapify(priority_queue)
        previous_edge_dict = {tuple(edge_node_id): None for edge_node_id in self.edge_df[['edge_id', 'to_id']].values.tolist()}
        distances[init_prev_edge] = 0
        flag = False
        last_edge = None

        while priority_queue:
            current_distance, previous_edge, current_node = heapq.heappop(priority_queue)

            if flag and current_distance >= distances[last_edge]:
                break

            if current_distance > distances[previous_edge]:
                continue

            for neighbor, weight, edge_id in self.graph.get(current_node, []):
                if edge_id == self._get_reverse_edge(previous_edge):
                    continue
                
                distance = current_distance + weight

                if distance < distances[edge_id]:
                    distances[edge_id] = distance
                    previous_edge_dict[(edge_id, neighbor)] = (previous_edge, current_node)
                    if neighbor == end_node:
                        last_edge = edge_id
                        flag = True
                    else:
                        heapq.heappush(priority_queue, (distance, edge_id, neighbor))

        assert last_edge is not None, "No path found"
        path = []
        current_edge, current_node = last_edge, end_node
        while current_edge != init_prev_edge:
            path.append(current_edge)
            current_edge, current_node = previous_edge_dict[(current_edge, current_node)]
        assert current_node == start_node, "Start node is not the same"
        path = path[::-1]
        # Ensure there are no identical consecutive edges.
        # for i in range(len(path) - 1):
        #     assert path[i] != path[i + 1], f"Two same edges in a row in shortest path: {path}"

        # assert not self.have_u_turn(path), f"U-turn in shortest path\n{path}"

        return path, distances[last_edge]

    def have_u_turn(self, path):
        for i in range(len(path) - 1):
            if self.edge_df.loc[self.edge_df['edge_id'] == path[i], 'from_id'].values[0] == \
                    self.edge_df.loc[self.edge_df['edge_id'] == path[i + 1], 'to_id'].values[0]:
                return True
        return False

    # Wandering strategy so idle vehicles keep moving.
    def wander_from_node(self, start_node, prev_node=None):
        wander_path = []
        cur_node = start_node

        for _ in range(self.wander_steps):
            neighbors = self.graph.get(cur_node, None)
            assert neighbors is not None, "No neighbors found"
            assert len(neighbors) > 1, "Out degree <= 1"
            if len(neighbors) > 1:  # If multiple neighbors exist, avoid immediate backtracking.
                if prev_node is not None:
                    neighbors = [(node, weight, edge_id) 
                                for node, weight, edge_id in neighbors if node != prev_node]
            next_node, _, edge_id = random.choice(neighbors)
            wander_path.append(edge_id)
            prev_node = cur_node
            cur_node = next_node
        # for i in range(len(wander_path) - 1):
        #     assert wander_path[i] != wander_path[i + 1], "Two same edges in a row in wander path"
        # assert not self.have_u_turn(wander_path), f"U-turn in wander path\n{wander_path}"
        return wander_path

    def apply_route(self, start_edge, loc_lst, dest_lst):
        if start_edge is None:
            start_edge = self.edge_df.loc[self.edge_df["to_id"] == self.find_nearest_node(loc_lst[0], force_no_isolate=True), "edge_id"].values[0]
        start_edge = int(start_edge)
        path = [start_edge]
        prev_node = self.edge_df.loc[self.edge_df['edge_id'] == start_edge, 'from_id'].values[0]
        cur_node = self.edge_df.loc[self.edge_df['edge_id'] == start_edge, 'to_id'].values[0]
        distance = 0
        start_node_idcs = []
        end_node_idcs = []
        for loc in loc_lst:
            loc_node = self.find_nearest_node(loc, force_no_isolate=True)
            partial_path, partial_distance = self.shortest_path(cur_node, loc_node, init_prev_node=prev_node)
            distance = distance + partial_distance
            path.extend(partial_path)
            start_node_idcs.append(len(path))
            cur_node = loc_node
            prev_node = self.edge_df.loc[self.edge_df['edge_id'] == path[-1], 'from_id'].values[0]
        for dest in dest_lst:
            dest_node = self.find_nearest_node(dest, force_no_isolate=True)
            partial_path, partial_distance = self.shortest_path(cur_node, dest_node, init_prev_node=prev_node)
            distance = distance + partial_distance
            path.extend(partial_path)
            end_node_idcs.append(len(path))
            cur_node = dest_node
            prev_node = self.edge_df.loc[self.edge_df['edge_id'] == path[-1], 'from_id'].values[0]
        path.extend(self.wander_from_node(cur_node, prev_node=prev_node))
        assert None not in set(path), "None in path"

        # for i in range(len(path) - 1):
        #     assert path[i] != path[i + 1], "Two same edges in a row in route"
        assert path[0] == start_edge, "Start edge is not the same"
        # assert not self.have_u_turn(path), f"U-turn in route\n{path}"
        start_node_idcs = [len(path) - i for i in start_node_idcs]
        end_node_idcs = [len(path) - i for i in end_node_idcs]
        return path, distance, start_node_idcs, end_node_idcs

    def find_loc_from_edge(self, edge_id):
        to_node = self.edge_df.loc[self.edge_df['edge_id'] == edge_id, 'to_id'].values
        if len(to_node) == 0:
            return None  # Located at a node with out-degree <= 1.
        to_node = to_node[0]
        return self.node_df.loc[self.node_df['node_id'] == to_node, ['lat', 'lon']].values[0]

    def wander_from_edge(self, edge_id):
        edge_id = int(edge_id)
        to_node = self.edge_df.loc[self.edge_df['edge_id'] == edge_id, 'to_id'].values[0]
        from_node = self.edge_df.loc[self.edge_df['edge_id'] == edge_id, 'from_id'].values[0]
        wander_path = [edge_id] + self.wander_from_node(to_node, prev_node=from_node)
        # assert not self.have_u_turn(wander_path), f"U-turn in wander path\n{wander_path}"
        return wander_path

    def _check_connectivity(self, path):
        for i in range(len(path) - 1):
            if self.edge_df.loc[self.edge_df['edge_id'] == path[i], 'to_id'].values[0] != \
                    self.edge_df.loc[self.edge_df['edge_id'] == path[i + 1], 'from_id'].values[0]:
                print(f"Connectivity error: {path[i]} -> {path[i + 1]}")
                print(i)
                return False
        return True

    def get_coord_from_node(self, node_id):
        return list(self.node_df.loc[self.node_df['node_id'] == node_id, ['lat', 'lon']].values[0])

    def get_coord_from_edge(self, edge_id, from_node):
        mode = 'from_id' if from_node else 'to_id'
        node = self.edge_df.loc[self.edge_df['edge_id'] == edge_id, mode].values[0]
        return self.get_coord_from_node(node)

    def calc_path_distance(self, path):
        distance = 0
        for edge_id in path:
            from_id = self.edge_df.loc[self.edge_df["edge_id"] == edge_id, "from_id"].values[0]
            flag = False
            for end_node, weight, _edge_id in self.graph.get(from_id):
                if _edge_id == edge_id:
                    distance += weight
                    flag = True
                    break
            assert flag, f"Edge {edge_id} not found in graph"
        return distance
