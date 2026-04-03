# Task 5: Minimal ROS2 A* Planner

This submission is meant to be built and run on Ubuntu 24.04 with ROS 2 Jazzy.

## Best macOS workaround

The most practical workaround on this Mac is UTM running Ubuntu 24.04 Desktop. Docker can still be used for headless Linux builds with the official `ros:jazzy` image.

## Ubuntu 24.04 + ROS2 Jazzy setup

Install ROS2 Jazzy:

```bash
sudo apt update
sudo apt install ros-jazzy-desktop
source /opt/ros/jazzy/setup.bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```

Install build tools and required ROS packages:

```bash
sudo apt install build-essential cmake git
sudo apt install python3-colcon-common-extensions
sudo apt install ros-jazzy-nav-msgs
sudo apt install ros-jazzy-geometry-msgs
sudo apt install ros-jazzy-std-msgs
sudo apt install ros-jazzy-rviz2
sudo apt install ros-jazzy-nav2-map-server
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

Do the full copy/build/setup in one command:

```bash
cd /path/to/task-phase-5
./setup_linux_vm.sh
```

Run the planner task with 4 scripts in 4 terminals:

```bash
./run_map_server.sh
```

```bash
./activate_map_server.sh
```

```bash
./run_astar_planner.sh
```

```bash
./run_rviz_task5.sh
```

Use another map:

```bash
./run_map_server.sh map_lv2.yaml
```

In RViz add:

- `Map` on `/map`
- `Path` on `/path`

## Chat Interface Package

Task 5 also includes a second ROS 2 C++ package:

- `chat_interface`

It contains:

- `talker` — publishes terminal input to `/chat`
- `listener` — subscribes to `/chat` and prints received messages

Topic and type:

- `/chat`
- `std_msgs/msg/String`

Run the chat nodes in two terminals:

```bash
./run_chat_listener.sh
```

```bash
./run_chat_talker.sh
```

## Scripts for Ubuntu VM

If you are using Ubuntu in UTM, you can avoid typing the full setup manually.

Run once:

```bash
cd /path/to/task-phase-5
chmod +x *.sh
./setup_linux_vm.sh
```

After that, use the individual run scripts for planner and chat.

## Build on macOS using Docker

This repository includes a helper script that builds both packages in a Linux ROS2 Jazzy container:

```bash
cd /path/to/tp5
./build_in_docker.sh
```

What it does:

- uses `ros:jazzy`
- installs the required build packages
- copies `astar_planner` and `chat_interface` into a temporary ROS2 workspace in the container
- runs `colcon build --packages-select astar_planner chat_interface`

## Theory note

A* is not ideal for maze solving when the full environment is unknown. It depends on a known map, keeps extra memory for open and closed sets, and is less practical than simpler exploration methods such as DFS or wall-following for real-time maze traversal.
