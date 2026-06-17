# av-sensor-sim

A ROS2-based AV sensor simulation pipeline that replays real lidar data from the nuScenes dataset, runs obstacle detection, and evaluates against ground truth annotations — demonstrating the core architecture used in autonomous vehicle stacks.

## What it does

1. **Lidar Publisher** — reads lidar point clouds from nuScenes mini split and publishes them to `/lidar/points` at 10Hz, simulating a live sensor stream
2. **Obstacle Detector** — subscribes to `/lidar/points`, runs voxel downsampling + DBSCAN clustering, and publishes detected obstacles as colored sphere markers (`/obstacles/markers`) and text alerts (`/obstacles/alerts`) for anything within 10m
3. **Offline Evaluator** — runs the same detection pipeline against nuScenes ground truth 3D bounding boxes, computes precision/recall/F1 using BEV IoU + Hungarian matching

## Architecture

```
nuScenes drive log
      ↓
lidar_publisher (ROS2 node)
      ↓ /lidar/points (PointCloud2, 10Hz)
obstacle_detector (ROS2 node)
      ↓ /obstacles/markers (MarkerArray)   ← visualized in Foxglove Studio
      ↓ /obstacles/alerts  (String)        ← text alert for <10m objects
```

## Detection pipeline

Each lidar frame goes through:

1. **Ground filter** — remove points below z = -1.0m (ground plane)
2. **Near filter** — remove points within 2.5m horizontal (car body returns)
3. **Voxel downsample** — collapse 30,000+ raw points into ~4,000 uniform 0.3m voxels
4. **DBSCAN** — cluster voxels into candidate objects (ε=1.5m, min_samples=5)
5. **Size filter** — drop clusters with <5 or >2000 voxels (noise and walls)
6. **Alert** — publish sphere marker per cluster; red if centroid <10m, green otherwise

## Evaluation results

Evaluated against nuScenes ground truth annotations for `vehicle.car`, `human.pedestrian.adult`, and `vehicle.bicycle` using 2D Bird's Eye View IoU (≥0.5 threshold) and Hungarian optimal matching across 404 frames.

| Range   | Precision | Recall | F1    |   TP |    FP |    FN |
|---------|-----------|--------|-------|------|-------|-------|
| 10m     | 0.143     | 0.200  | 0.167 |  192 |  1153 |   767 |
| 20m     | 0.068     | 0.106  | 0.083 |  374 |  5127 |  3168 |
| 30m     | 0.037     | 0.063  | 0.046 |  423 | 11078 |  6333 |
| 50m     | 0.017     | 0.043  | 0.025 |  439 | 24864 |  9803 |
| all     | 0.013     | 0.035  | 0.019 |  439 | 34349 | 12188 |

**Key observations:**

- **Precision drops with range** — DBSCAN clusters everything (buildings, walls, trees), generating ~100 FPs per frame beyond 20m. A category classifier on top of DBSCAN would fix this.
- **Recall drops with range** — distant objects have too few lidar returns to form a cluster. TP count plateaus at 439 past 50m: the detector is blind beyond that range.
- **At 10m the numbers are reasonable** — dense near-field returns make clusters reliable, and fewer background objects compete for matches.

## Why this matters

This pipeline mirrors what AV simulation engineers build at companies like Applied Intuition and Waymo:
- Real sensor data replayed through a live ROS2 pub/sub stream
- Modular nodes that can be swapped independently (e.g. replace the lidar publisher with a 3DGS synthetic renderer for novel viewpoints)
- Standard message types (`sensor_msgs/PointCloud2`) compatible with the broader ROS2 ecosystem
- Offline evaluation against a real annotated dataset using industry-standard metrics

## Setup

### Prerequisites
- Docker
- nuScenes mini split downloaded to `data/nuscenes/`

### Run live pipeline

```bash
# Start both nodes (Mac: network_mode host doesn't work, run in one container)
docker compose up

# Or manually — Terminal 1
docker run -it \
  -v $(pwd)/src:/av-sensor-sim/src \
  -v $(pwd)/data/nuscenes:/data/nuscenes \
  osrf/ros:humble-desktop \
  bash -c "source /opt/ros/humble/setup.bash && pip install nuscenes-devkit -q && python3 /av-sensor-sim/src/lidar_publisher.py"

# Terminal 2 — exec into the same container
docker exec -it <container_id> bash -c \
  "source /opt/ros/humble/setup.bash && pip install scikit-learn -q && python3 /av-sensor-sim/src/obstacle_detector.py"
```

Open [Foxglove Studio](https://foxglove.dev) and connect to `ws://localhost:8765` to visualize the point cloud and obstacle markers.

### Run offline evaluation

```bash
# Install deps outside Docker (uses nuScenes .venv)
pip install nuscenes-devkit scikit-learn scipy pyquaternion

# Single range
python3 src/evaluate.py --range 30

# Comparison table across multiple ranges
python3 src/evaluate.py --ranges 10 20 30 50

# All ranges (no flag)
python3 src/evaluate.py
```

### Debug
```bash
ros2 topic list           # check active topics
ros2 topic hz /lidar/points   # verify publish rate
ros2 topic echo /obstacles/alerts  # inspect messages
```

## Roadmap

- [x] Lidar publisher — replay nuScenes point clouds over ROS2
- [x] Obstacle detector — voxel downsample + DBSCAN + Foxglove markers
- [x] Offline evaluator — BEV IoU + Hungarian matching vs nuScenes GT
- [ ] Replace lidar publisher with 3DGS synthetic renderer for novel viewpoints
- [ ] Add camera image publisher alongside lidar
- [ ] Project lidar points onto camera frame (sensor fusion)

## Data

Uses [nuScenes mini split](https://www.nuscenes.org) — a standard AV dataset with lidar, camera, and radar data from real urban drives.
