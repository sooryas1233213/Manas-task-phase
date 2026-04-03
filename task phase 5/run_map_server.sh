#!/usr/bin/env bash
set -euo pipefail

MAP_NAME="${1:-map_basic.yaml}"
WS_DIR="${HOME}/ros2_ws"
MAP_PATH="${WS_DIR}/maps/${MAP_NAME}"

if [[ ! -f "${MAP_PATH}" ]]; then
  echo "Map YAML not found: ${MAP_PATH}"
  echo "Available maps:"
  ls -1 "${WS_DIR}/maps"/*.yaml 2>/dev/null || true
  exit 1
fi

source /opt/ros/jazzy/setup.bash
source "${WS_DIR}/install/setup.bash"
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:="${MAP_PATH}"
