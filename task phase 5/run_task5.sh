#!/usr/bin/env bash
set -euo pipefail

MAP_NAME="${1:-map_basic.yaml}"
WS_DIR="${HOME}/ros2_ws"
MAP_PATH="${WS_DIR}/maps/${MAP_NAME}"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="${WS_DIR}/install/setup.bash"

if [[ ! -f "${MAP_PATH}" ]]; then
  echo "Map YAML not found: ${MAP_PATH}"
  echo "Available maps:"
  ls -1 "${WS_DIR}/maps"/*.yaml 2>/dev/null || true
  exit 1
fi

source "${ROS_SETUP}"
source "${WS_SETUP}"

cleanup() {
  jobs -p | xargs -r kill >/dev/null 2>&1 || true
}
trap cleanup EXIT

ros2 run nav2_map_server map_server --ros-args -p yaml_filename:="${MAP_PATH}" &
MAP_SERVER_PID=$!
sleep 3

ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate

ros2 run astar_planner astar_planner_node &
PLANNER_PID=$!
sleep 2

echo "map_server pid: ${MAP_SERVER_PID}"
echo "planner pid: ${PLANNER_PID}"
echo "Launching RViz with ${MAP_NAME}"

rviz2
