#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source /opt/task8/scripts/task8_ros.sh

RVIZ_CONFIG="/opt/task8/rviz/task8_humble.rviz"
RVIZ_LOG="${TASK8_RVIZ_LOG:-/tmp/task8-rviz.log}"
RESTART=0

while (( "$#" )); do
  case "$1" in
    --restart)
      RESTART=1
      shift
      ;;
    --config)
      if [[ "$#" -lt 2 ]]; then
        echo "Missing value for --config."
        exit 1
      fi
      RVIZ_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ! -f "${RVIZ_CONFIG}" ]]; then
  echo "RViz config not found: ${RVIZ_CONFIG}"
  exit 1
fi

if (( RESTART )); then
  pkill -x rviz2 >/dev/null 2>&1 || true
  sleep 1
fi

if pgrep -x rviz2 >/dev/null 2>&1; then
  echo "rviz2 is already running."
  exit 0
fi

nohup rviz2 -d "${RVIZ_CONFIG}" >"${RVIZ_LOG}" 2>&1 &
sleep 3

if ! pgrep -x rviz2 >/dev/null 2>&1; then
  echo "rviz2 failed to start."
  tail -n 50 "${RVIZ_LOG}" || true
  exit 1
fi

echo "rviz2 started. Log: ${RVIZ_LOG}"
