# Robotaxi

Code of shared autonomous taxis
(robotaxis) on urban traffic. Given a city road network and an
origin-destination (OD) demand, the pipeline produces individual vehicle
trips, optionally re-expresses them as pooled robotaxi rides, and replays both
scenarios in the [CBEngine](https://cblab-documentation.readthedocs.io/en/latest/index.html) traffic simulator for comparison.

## Repository layout

```
align_id.py               # Extract node/edge tables from roadnet.txt
assign_volumn.py          # Convert detector flow data into OD requests
gen_private_car_route.py  # Per-trip routing for the private-car baseline
gen_robotaxi_route.py     # Cluster trips into shared robotaxi routes
main.py                   # Run the CBEngine simulation
robotaxi/
  dispatch.py             # Routing, matching and clustering primitives
  tracer.py               # Per-vehicle bookkeeping during simulation
  calc_private_detail.py  # Build the trace JSON consumed by main.py
data/<city>/              # Per-city inputs and generated artefacts
```

## Per-city inputs

Place the following files under `data/<city>/`:

| File | Purpose |
| --- | --- |
| `roadnet.txt` | CBEngine road network. |
| `private_car_config.cfg`, `robotaxi_config.cfg` | CBEngine configuration files. |
| `od_requests.json` | OD demand list. May be regenerated from raw detector data via `assign_volumn.py`. |

## Pipeline

The steps below assume `roadnet.txt` and `od_requests.json` are available for a
target city (`basel` is used in the examples).

### 1. Align node and edge identifiers

```bash
python align_id.py --data_dir data/basel
```

Produces `align_node.csv` and `align_edge.csv`.

### 2. (Optional) Generate OD requests from detector data

If only raw UTD19-style traffic counts are available, OD requests can be
produced by greedily attributing observed flow to shortest paths:

```bash
python assign_volumn.py --root_dir data --city basel --scale 1.0
```

Skip this step when an `od_requests.json` file is already provided.

### 3. Route the private-car baseline

```bash
python gen_private_car_route.py \
    --request_file    data/basel/od_requests.json \
    --align_files_dir data/basel \
    --output_file     data/basel/private_car_route.txt
```

Only requests in the 8&ndash;10 a.m. window are retained.

### 4. Build the robotaxi route file

Cluster nearby private-car trips that start within the same waiting window and
re-route each cluster as a shared ride. A `.json` sidecar with per-trip
pickup and drop-off indices is produced alongside the route file.

```bash
python gen_robotaxi_route.py \
    --private_car_route_file data/basel/private_car_route.txt \
    --align_files_dir        data/basel \
    --output_file            data/basel/robotaxi_route.txt \
    --radius                 200
```

### 5. Build the trace JSON for the private-car run

```bash
python -m robotaxi.calc_private_detail \
    --align_files_dir        data/basel \
    --private_car_route_file data/basel/private_car_route.txt \
    --output_file            data/basel/private_car_route.txt.json
```

### 6. Run the simulation

Each scenario uses its own configuration file, which references the matching
route file:

```bash
# Baseline
python main.py \
    --trace_file   data/basel/private_car_route.txt.json \
    --roadnet_file data/basel/roadnet.txt \
    --cfg_file     data/basel/private_car_config.cfg \
    --logging      log/basel_private.log

# Robotaxi
python main.py \
    --trace_file   data/basel/robotaxi_route.txt.json \
    --roadnet_file data/basel/roadnet.txt \
    --cfg_file     data/basel/robotaxi_config.cfg \
    --logging      log/basel_robotaxi.log
```

A summary (average travel time, peak vehicle count) is written to
`result/<city>.txt`, and a per-speed histogram to
`result/<city>_<trace_file>`.