#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${HOME}/ros2_ws"
PKG_SRC_DIR="${WS_DIR}/src/astar_planner"
MAPS_DST_DIR="${WS_DIR}/maps"
ROS_SETUP="/opt/ros/humble/setup.bash"

if [[ ! -f /etc/os-release ]]; then
  echo "This script must be run on Ubuntu Linux."
  exit 1
fi

if ! grep -q 'Ubuntu' /etc/os-release; then
  echo "This script is intended for Ubuntu."
  exit 1
fi

sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release
sudo add-apt-repository universe -y

if [[ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
fi

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME}") main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  build-essential \
  cmake \
  git \
  python3-colcon-common-extensions \
  ros-humble-nav-msgs \
  ros-humble-geometry-msgs \
  ros-humble-rviz2 \
  ros-humble-nav2-map-server

if ! grep -Fq "source ${ROS_SETUP}" "${HOME}/.bashrc"; then
  echo "source ${ROS_SETUP}" >> "${HOME}/.bashrc"
fi

mkdir -p "${WS_DIR}/src"
mkdir -p "${MAPS_DST_DIR}"
rm -rf "${PKG_SRC_DIR}"
cp -r "${SCRIPT_DIR}/astar_planner" "${PKG_SRC_DIR}"
cp -f "${SCRIPT_DIR}/maps/"* "${MAPS_DST_DIR}/"

source "${ROS_SETUP}"
cd "${WS_DIR}"
colcon build --packages-select astar_planner

if ! grep -Fq "source ${WS_DIR}/install/setup.bash" "${HOME}/.bashrc"; then
  echo "source ${WS_DIR}/install/setup.bash" >> "${HOME}/.bashrc"
fi

echo
echo "Setup complete."
echo "Package copied to: ${PKG_SRC_DIR}"
echo "Maps copied to: ${MAPS_DST_DIR}"
echo "Next run: ${SCRIPT_DIR}/run_task5.sh"
