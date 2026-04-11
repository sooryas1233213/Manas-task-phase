#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/common.sh"

START_TIMEOUT="${TASK8_START_TIMEOUT:-240}"
REBUILD="${TASK8_PHASE0_REBUILD:-1}"
VO_LOG="${TASK8_VO_LOG:-/tmp/task8-monocular-vo.log}"
PHASE0_RVIZ_CONFIG="/project/config/rviz/task8_phase0_vo.rviz"

echo "Starting Task 8 stack for monocular VO..."
start_stack "${START_TIMEOUT}" "${REBUILD}"

echo "Building the Task 8 workspace..."
"${SCRIPT_DIR}/build_ws.sh"

echo "Launching VO RViz config..."
docker_compose exec -T "${SERVICE_NAME}" bash -lc \
  "/opt/task8/scripts/run_rviz.sh --restart --config '${PHASE0_RVIZ_CONFIG}'"

container="$(container_id)"
if [[ -z "${container}" ]]; then
  echo "Could not resolve the Task 8 container ID."
  exit 1
fi

echo "Starting monocular_vo launch..."
docker exec "${container}" bash -lc "
  python3 - <<'PY'
import os
import signal

patterns = [
    '/opt/ros/humble/bin/ros2 launch monocular_vo vo.launch.py',
    '/ws/install/monocular_vo/lib/monocular_vo/video_bridge',
    '/ws/install/monocular_vo/lib/monocular_vo/vo_node',
    '/opt/ros/humble/lib/tf2_ros/static_transform_publisher 0.0 0.0 0.0 0 0 0 1 base_link camera_link',
    '/opt/ros/humble/lib/tf2_ros/static_transform_publisher 0 0 0 -0.5 0.5 -0.5 0.5 camera_link camera_optical_frame',
]
exclude = {os.getpid(), os.getppid()}
targets = []
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    pid_int = int(pid)
    if pid_int in exclude:
        continue
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as fh:
            cmdline = fh.read().replace(b'\\x00', b' ').decode(errors='ignore')
    except OSError:
        continue
    if any(pattern in cmdline for pattern in patterns):
        targets.append(pid_int)
for pid_int in sorted(set(targets)):
    try:
        os.kill(pid_int, signal.SIGTERM)
    except ProcessLookupError:
        pass
PY
"
docker exec -d "${container}" bash -lc \
  "source /opt/task8/scripts/task8_ros.sh && exec ros2 launch monocular_vo vo.launch.py >'${VO_LOG}' 2>&1"

echo "Waiting for VO nodes..."
started_at="$(date +%s)"
while true; do
  if docker_compose exec -T "${SERVICE_NAME}" bash -lc "
    source /opt/task8/scripts/task8_ros.sh
    ros2 node list 2>/dev/null | grep -Fx '/video_bridge' >/dev/null &&
    ros2 node list 2>/dev/null | grep -Fx '/vo_node' >/dev/null
  "; then
    break
  fi

  if (( $(date +%s) - started_at >= START_TIMEOUT )); then
    echo "Timed out waiting for VO nodes to start."
    docker_compose exec -T "${SERVICE_NAME}" bash -lc "tail -n 120 '${VO_LOG}' || true"
    exit 1
  fi

  sleep 2
done

echo
echo "Monocular VO is running."
echo "Launch log: ${VO_LOG}"
echo "${TASK8_URL}"

if [[ "${OPEN_BROWSER:-1}" == "1" ]] && command -v open >/dev/null 2>&1; then
  open "${TASK8_URL}" >/dev/null 2>&1 || true
fi
