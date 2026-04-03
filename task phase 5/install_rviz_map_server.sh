#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y ros-jazzy-rviz2 ros-jazzy-nav2-map-server
source /opt/ros/jazzy/setup.bash
echo "RViz2 and nav2_map_server installation complete."
