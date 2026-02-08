import json
from copy import deepcopy

class Tracer:
    def __init__(self, detail_file: str, max_speed: int):
        with open(detail_file, "r") as f:
            self.details = json.load(f)
        self.max_speed = max_speed
        self.speed_hist = [0 for _ in range(self.max_speed+1)]
        # Initialize vehicle speed histogram
        # key: vehicle_id, value: [0, 0, ..., 0] (length max_speed+1)
        # where index i represents the count of speed i
        self.vihecle_speed_hist = {}
        self.already_tracing_vehicle_ids = set()
        self.active_vehicle_ids = set()
        self.tracer_vehicle_infos = {}
        self.preprocess_details()

    def preprocess_details(self):
        self.no_tracing_details = deepcopy(self.details)

    def trace(self, engine):
        current_time = engine.get_current_time()
        no_yet_trace_vehicles = set(engine.get_vehicles()) - self.already_tracing_vehicle_ids
        for vehicle_id in no_yet_trace_vehicles:
            flag = False
            for detail in self.no_tracing_details:
                # print(current_time, detail["time"], current_time == detail["time"])
                # print(detail["route"], list(map(int, engine.get_vehicle_info(vehicle_id)["route"])))
                if current_time == detail["time"]+1 and detail["route"] == list(map(int, engine.get_vehicle_info(vehicle_id)["route"])):
                    self.already_tracing_vehicle_ids.add(vehicle_id)
                    detail.update({
                        "start_time": current_time,
                        "cur_people": 0,
                        "pickup_people_time_lst": [],
                        "dropoff_people_time_lst": [],
                        "pickup_all": False,
                        "tot_customer": len(detail["start_idcs"])
                    })
                    self.tracer_vehicle_infos[vehicle_id] = detail
                    self.active_vehicle_ids.add(vehicle_id)
                    self.no_tracing_details.remove(detail)
                    flag = True
                    break
            assert flag, f"Vehicle {vehicle_id} not found in details\n{engine.get_vehicle_info(vehicle_id)}"

        to_remove = set()
        for vehicle_id in self.active_vehicle_ids:
            info: list = self.tracer_vehicle_infos[vehicle_id]
            engine_info = engine.get_vehicle_info(vehicle_id)
            if not engine_info:
                engine_info_route = []
            else:
                engine_info_route = list(map(int, engine_info["route"]))
            
            while (not info["pickup_all"]) and info["cur_people"] < info["tot_customer"]:
                if len(engine_info_route) <= info["start_idcs"][info["cur_people"]]:
                    info["cur_people"] += 1
                    info["pickup_people_time_lst"].append(current_time)
                    if info["cur_people"] == info["tot_customer"]:
                        info["pickup_all"] = True
                else:
                    break
            
            while info["pickup_all"] and info["cur_people"] > 0:
                idx = info["tot_customer"] - info["cur_people"]
                if len(engine_info_route) <= info["end_idcs"][idx]:
                    info["cur_people"] -= 1
                    info["dropoff_people_time_lst"].append(current_time)
                else:
                    break

            if not engine_info:
                info["end_time"] = current_time
                to_remove.add(vehicle_id)
        
        for vehicle_id in to_remove:
            self.active_vehicle_ids.remove(vehicle_id)

        self.get_speed_hist(engine)
        self.get_vehicle_speed_hist(engine)

    def get_speed_hist(self, engine):
        speeds = engine.get_vehicle_speed()
        speed_values = [v for k, v in speeds.items() if k in self.active_vehicle_ids]
        for speed in speed_values:
            self.speed_hist[max(0, min(round(speed), self.max_speed))] += 1
    
    def get_vehicle_speed_hist(self, engine):
        speeds = engine.get_vehicle_speed()
        for vehicle_id, speed in speeds.items():
            if vehicle_id in self.active_vehicle_ids:
                if vehicle_id not in self.vihecle_speed_hist:
                    self.vihecle_speed_hist[vehicle_id] = [0 for _ in range(self.max_speed+1)]
                self.vihecle_speed_hist[vehicle_id][max(0, min(round(speed), self.max_speed))] += 1

    @staticmethod
    def calc_fuel(distance: float):
        # tot_distance: m
        # return: L
        return distance * 0.001 * 0.01 * 8
    ## Fuel consumption assumption: 8 L/100 km

    def calc_carbon_emission(self):
        
        def getCO2(speed: int):
            # speed: m/s
            speed_kmh = speed * 3.6
            g_CO2_per_km = 209 - 3.12*speed_kmh + 0.0258*speed_kmh**2 + (1470/speed_kmh if speed_kmh >= 10 else 147)
            kg_CO2_per_m = g_CO2_per_km * 0.001 * 0.001
            emission_kg_per_s = speed * kg_CO2_per_m
            return emission_kg_per_s
        
        emission = 0
        for speed, count in enumerate(self.speed_hist):
            emission += getCO2(speed) * count
        return emission

    def carbon_emission_fuel(self, fuel):
        # Calculate carbon emission based on fuel consumption
        # 1L gasoline produces about 2.2601 kg of CO2
        return fuel * 2.2601

    def calc_and_print_info(self, logger=None):
        tot_travel_time = 0
        tot_trip = 0
        tot_trip_time = 0
        tot_people = 0
        tot_distance = 0
        tot_pickup_time = 0
        for vehicle_id, info in self.tracer_vehicle_infos.items():
            tot_trip += 1
            tot_people += info["tot_customer"]
            tot_trip_time += info.get("end_time", 28800) - info["start_time"]
            tot_distance += info["distance"]
            assert len(info["pickup_people_time_lst"]) == len(info["dropoff_people_time_lst"]) and len(info["pickup_people_time_lst"]) == info["tot_customer"],\
                f"vehicle:{vehicle_id} info: {info}"
            tot_travel_time += sum([info["dropoff_people_time_lst"][i] - info["pickup_people_time_lst"][i] for i in range(info["tot_customer"])])
            tot_pickup_time += sum([info["pickup_people_time_lst"][i] - info["start_time"] for i in range(info["tot_customer"])])
        avg_travel_time = tot_travel_time / tot_people
        avg_trip_time = tot_trip_time / tot_trip
        avg_trip_distance = tot_distance / tot_trip
        avg_people_per_trip = tot_people / tot_trip
        fuel = self.calc_fuel(tot_distance)
        co2_fuel = self.carbon_emission_fuel(fuel)
        carbon_emission = self.calc_carbon_emission()
        avg_speed_hist = [] # (vihecle_id, avg_speed, count)
        for vihecle_id, speed_hist in self.vihecle_speed_hist.items():
            avg_speed = sum([i * count for i, count in enumerate(speed_hist)]) / len(speed_hist) if sum(speed_hist) > 0 else -1
            avg_speed_hist.append((vihecle_id, avg_speed, len(speed_hist)))
        
        putout = lambda x: print(x) if logger is None else logger.info(x)
        putout(f"Total trips: {tot_trip}, Total people: {tot_people}, Total distance: {tot_distance}")
        putout(f"Avg travel time: {avg_travel_time}, Avg trip time: {avg_trip_time}, Avg trip distance: {avg_trip_distance}, Avg people per trip: {avg_people_per_trip}")
        putout(f"Avg pickup time: {tot_pickup_time / tot_people if tot_people > 0 else 0}")
        putout(f"Total fuel consumption: {fuel} L")
        putout(f"Total carbon emission from fuel: {co2_fuel} kg")
        putout(f"Total carbon emission: {carbon_emission} kg")
        putout(f"Avg speed for trips: {sum([i[2] for i in avg_speed_hist]) / len(avg_speed_hist) if len(avg_speed_hist) > 0 else -1} m/s")
