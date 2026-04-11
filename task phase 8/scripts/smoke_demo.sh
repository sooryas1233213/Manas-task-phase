#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source /opt/task8/scripts/task8_ros.sh

STATIC_TF_LOG="/tmp/task8-static-tf.log"
MARKER_LOG="/tmp/task8-marker.log"
MARKER_MESSAGE="$(cat <<'EOF'
{header: {frame_id: map}, ns: task8, id: 0, type: 2, action: 0, pose: {position: {x: 0.0, y: 0.0, z: 0.5}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, scale: {x: 0.5, y: 0.5, z: 0.5}, color: {r: 0.15, g: 0.85, b: 0.2, a: 1.0}}
EOF
)"

wait_for_topic() {
  local topic_name="$1"
  local timeout_secs="$2"
  local started_at

  started_at="$(date +%s)"

  while true; do
    if ros2 topic list 2>/dev/null | grep -Fx "${topic_name}" >/dev/null; then
      return 0
    fi

    if (( $(date +%s) - started_at >= timeout_secs )); then
      echo "Timed out waiting for topic ${topic_name}."
      return 1
    fi

    sleep 1
  done
}

pkill -f 'static_transform_publisher 0 0 0 0 0 0 map marker_frame' >/dev/null 2>&1 || true
pkill -f '/visualization_marker visualization_msgs/msg/Marker' >/dev/null 2>&1 || true

nohup ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map marker_frame >"${STATIC_TF_LOG}" 2>&1 &
nohup ros2 topic pub --rate 1 /visualization_marker visualization_msgs/msg/Marker "${MARKER_MESSAGE}" >"${MARKER_LOG}" 2>&1 &

wait_for_topic /tf_static 15
wait_for_topic /visualization_marker 15

/opt/task8/scripts/run_rviz.sh --restart

if ! pgrep -x rviz2 >/dev/null 2>&1; then
  echo "rviz2 is not running after smoke test startup."
  exit 1
fi

if ! pgrep -f 'static_transform_publisher 0 0 0 0 0 0 map marker_frame' >/dev/null 2>&1; then
  echo "Static transform publisher is not running."
  exit 1
fi

if ! pgrep -f '/visualization_marker visualization_msgs/msg/Marker' >/dev/null 2>&1; then
  echo "Marker publisher is not running."
  exit 1
fi

echo "Smoke demo is running."
echo "Topics:"
ros2 topic list | grep -E '^/tf_static$|^/visualization_marker$'
