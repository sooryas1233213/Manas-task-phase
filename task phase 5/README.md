# Task 5: Minimal ROS2 A* Planner

This submission is meant to be built and run on Linux with ROS 2 Humble.

## Best macOS workaround

The most practical workaround on this Mac is Docker Desktop with the official `ros:humble` Linux image. This keeps the build on Linux while letting you work from macOS. If you need native RViz GUI rendering and the Docker GUI setup becomes annoying, use a UTM VM running Ubuntu 22.04 as the fallback.

## Ubuntu 22.04 + ROS2 Humble setup

Install ROS2 Humble:

```bash
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

Install build tools and required ROS packages:

```bash
sudo apt install build-essential cmake git
sudo apt install python3-colcon-common-extensions
sudo apt install ros-humble-nav-msgs
sudo apt install ros-humble-geometry-msgs
sudo apt install ros-humble-rviz2
sudo apt install ros-humble-nav2-map-server
```

## Workspace setup on Linux or inside Ubuntu VM

Create the ROS2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

Copy this package into the workspace:

```bash
cp -r /path/to/tp5/astar_planner ~/ros2_ws/src/
```

Download the provided map folder and place it inside:

```bash
~/ros2_ws/src/
```

## Build and run on Linux

Build the planner package:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

Run the map server:

```bash
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/absolute/path/to/map.yaml
```

Run the planner node:

```bash
ros2 run astar_planner astar_planner_node
```

Open RViz:

```bash
rviz2
```

In RViz add:

- `Map` on `/map`
- `Path` on `/path`

## Scripts for Ubuntu VM

If you are using Ubuntu in UTM, you can avoid typing the full setup manually.

Run once:

```bash
cd /path/to/task-phase-5
chmod +x setup_linux_vm.sh run_task5.sh
./setup_linux_vm.sh
```

Run the full demo later:

```bash
cd /path/to/task-phase-5
./run_task5.sh
```

Use a different map:

```bash
./run_task5.sh map_lv2.yaml
```

## Build on macOS using Docker

This repository includes a helper script that builds the package in a Linux ROS2 Humble container:

```bash
cd /path/to/tp5
./build_in_docker.sh
```

What it does:

- uses `ros:humble`
- installs the required build packages
- copies `astar_planner` into a temporary ROS2 workspace in the container
- runs `colcon build --packages-select astar_planner`

## Theory note

A* is not ideal for maze solving when the full environment is unknown. It depends on a known map, keeps extra memory for open and closed sets, and is less practical than simpler exploration methods such as DFS or wall-following for real-time maze traversal.
