#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate
