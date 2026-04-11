#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/common.sh"

START_TIMEOUT="${TASK8_START_TIMEOUT:-240}"

echo "Building and starting the Task 8 ROS desktop container..."
start_stack "${START_TIMEOUT}" 1

echo "Launching RViz inside the container..."
docker_compose exec -T "${SERVICE_NAME}" bash -lc "/opt/task8/scripts/run_rviz.sh"

echo
echo "Task 8 desktop is ready:"
echo "${TASK8_URL}"

if [[ "${OPEN_BROWSER:-1}" == "1" ]] && command -v open >/dev/null 2>&1; then
  open "${TASK8_URL}" >/dev/null 2>&1 || true
fi
