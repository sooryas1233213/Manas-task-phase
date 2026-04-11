#!/usr/bin/env bash

if [[ -n "${TASK8_ROS_ENV_SOURCED:-}" ]]; then
  return 0 2>/dev/null || true
fi

export TASK8_ROS_ENV_SOURCED=1
export DISPLAY="${DISPLAY:-:1}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-ros}"
export NO_AT_BRIDGE="${NO_AT_BRIDGE:-1}"

mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true

had_nounset=0
if [[ $- == *u* ]]; then
  had_nounset=1
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  if (( had_nounset )); then
    set +u
  fi
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  if (( had_nounset )); then
    set -u
  fi
fi

if [[ -f /ws/install/setup.bash ]]; then
  if (( had_nounset )); then
    set +u
  fi
  # shellcheck disable=SC1091
  source /ws/install/setup.bash
  if (( had_nounset )); then
    set -u
  fi
fi

unset had_nounset
