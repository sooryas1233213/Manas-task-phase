#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)

docker run --rm   -v "$SCRIPT_DIR:/ws"   -w /ws   ros:jazzy   bash -lc '
    set -e
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y       build-essential       cmake       python3-colcon-common-extensions       ros-jazzy-nav-msgs       ros-jazzy-geometry-msgs       ros-jazzy-std-msgs
    source /opt/ros/jazzy/setup.bash
    mkdir -p /tmp/ros2_ws/src
    rm -rf /tmp/ros2_ws/src/astar_planner
    rm -rf /tmp/ros2_ws/src/chat_interface
    cp -r /ws/astar_planner /tmp/ros2_ws/src/
    cp -r /ws/chat_interface /tmp/ros2_ws/src/
    cd /tmp/ros2_ws
    colcon build --packages-select astar_planner chat_interface
  '
