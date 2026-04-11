#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/common.sh"

START_TIMEOUT="${TASK8_START_TIMEOUT:-240}"

start_stack "${START_TIMEOUT}"

docker_compose exec -T "${SERVICE_NAME}" bash -lc '
  source /opt/task8/scripts/task8_ros.sh

  if ! find /ws/src \( -name package.xml -o -name setup.py \) -print -quit | grep -q .; then
    echo "No ROS packages found under /ws/src."
    exit 0
  fi

  cd /ws
  colcon build --symlink-install
'
