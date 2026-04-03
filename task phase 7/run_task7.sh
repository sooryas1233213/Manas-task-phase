#!/usr/bin/env bash
set -euo pipefail

ROS_SETUP="/opt/ros/jazzy/setup.bash"
WS_DIR="${HOME}/turtlebot3_ws"
WS_SETUP="${WS_DIR}/install/setup.bash"
LOG_DIR="${TMPDIR:-/tmp}/task7_nav2_demo"
GAZEBO_LOG="${LOG_DIR}/gazebo.log"

source_script_safely() {
  local script_path="$1"
  set +u
  # shellcheck disable=SC1090
  source "${script_path}"
  set -u
}

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS 2 Jazzy was not found at ${ROS_SETUP}."
  echo "Run ./setup_linux_vm.sh first on Ubuntu 24.04."
  exit 1
fi

if [[ ! -f "${WS_SETUP}" ]]; then
  echo "Workspace setup file was not found at ${WS_SETUP}."
  echo "Run ./setup_linux_vm.sh first so ~/turtlebot3_ws gets built."
  exit 1
fi

source_script_safely "${ROS_SETUP}"
source_script_safely "${WS_SETUP}"

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

if ! ros2 pkg prefix task7_nav2_demo >/dev/null 2>&1; then
  echo "task7_nav2_demo is not available in the sourced workspace."
  echo "Run ./setup_linux_vm.sh first."
  exit 1
fi

mkdir -p "${LOG_DIR}"

cleanup() {
  if [[ -n "${GAZEBO_PID:-}" ]] && kill -0 "${GAZEBO_PID}" 2>/dev/null; then
    kill "${GAZEBO_PID}" 2>/dev/null || true
    wait "${GAZEBO_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Launching TurtleBot3 Gazebo world in the background..."
echo "Gazebo log: ${GAZEBO_LOG}"

ros2 launch task7_nav2_demo gazebo_world.launch.py >"${GAZEBO_LOG}" 2>&1 &
GAZEBO_PID=$!

sleep 8

if ! kill -0 "${GAZEBO_PID}" 2>/dev/null; then
  echo "Gazebo exited before Nav2 started."
  echo "Check ${GAZEBO_LOG} for details."
  exit 1
fi

echo "Launching Nav2 and RViz in the foreground..."
ros2 launch task7_nav2_demo nav2_rviz.launch.py
