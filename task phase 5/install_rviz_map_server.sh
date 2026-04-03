#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y ros-humble-rviz2 ros-humble-nav2-map-server
source /opt/ros/humble/setup.bash
echo "RViz2 and nav2_map_server installation complete."
