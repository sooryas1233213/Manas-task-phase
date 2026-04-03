#!/usr/bin/env bash
set -euo pipefail

WS_DIR="${HOME}/ros2_ws"

source /opt/ros/jazzy/setup.bash
source "${WS_DIR}/install/setup.bash"
ros2 run chat_interface talker
