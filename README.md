# av-sensor-sim

A ROS2-based AV sensor simulation pipeline that replays real lidar data from the nuScenes dataset and runs obstacle detection — demonstrating the core pub/sub architecture used in autonomous vehicle stacks.

## What it does

1. **Lidar Publisher** — reads lidar point clouds from nuScenes mini split and publishes them to `/lidar/points` at 10Hz, simulating a live sensor stream
2. **Obstacle Detector** — subscribes to `/lidar/points`, filters the ground plane, finds the closest object, and publishes alerts to `/obstacles/alerts` when anything is within 10 meters

## Architecture

```
nuScenes drive log
      ↓
lidar_publisher (ROS2 node)
      ↓ /lidar/points (PointCloud2, 10Hz)
obstacle_detector (ROS2 node)
      ↓ /obstacles/alerts (String)
```

## Why this matters

This pipeline mirrors what AV simulation engineers build at companies like Applied Intuition and Waymo:
- Real sensor data replayed through a live ROS2 stream
- Modular nodes that can be swapped independently (e.g. replace the lidar publisher with a 3DGS synthetic renderer)
- Standard message types (`sensor_msgs/PointCloud2`) compatible with the broader ROS2 ecosystem

## Setup

### Prerequisites
- Docker
- nuScenes mini split downloaded to `data/nuscenes/`

### Run

```bash
# Terminal 1 — lidar publisher
docker run -it \
  -v $(pwd)/src:/av-sensor-sim/src \
  -v $(pwd)/data/nuscenes:/data/nuscenes \
  --network host \
  osrf/ros:humble-desktop \
  bash -c "source /opt/ros/humble/setup.bash && pip install nuscenes-devkit -q && python3 /av-sensor-sim/src/lidar_publisher.py"

# Terminal 2 — obstacle detector
docker run -it \
  -v $(pwd)/src:/av-sensor-sim/src \
  --network host \
  osrf/ros:humble-desktop \
  bash -c "source /opt/ros/humble/setup.bash && python3 /av-sensor-sim/src/obstacle_detector.py"
```

### Debug
```bash
# Check active topics
ros2 topic list

# Verify publish rate
ros2 topic hz /lidar/points

# Inspect messages
ros2 topic echo /obstacles/alerts
```

## Roadmap

- [ ] Add camera image publisher alongside lidar
- [ ] Project lidar points onto camera frame (sensor fusion)
- [ ] Replace lidar publisher with 3DGS synthetic renderer for novel viewpoints
- [ ] Add RViz visualization

## Data

Uses [nuScenes mini split](https://www.nuscenes.org) — a standard AV dataset with lidar, camera, and radar data from real drives.
