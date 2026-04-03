#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${HOME}/turtlebot3_ws"
SRC_DIR="${WS_DIR}/src"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
TASK7_SRC="${SCRIPT_DIR}/task7_nav2_demo"
TASK7_DST="${SRC_DIR}/task7_nav2_demo"

if [[ ! -f /etc/os-release ]]; then
  echo "This setup script must be run on Ubuntu 24.04."
  exit 1
fi

. /etc/os-release

if [[ "${ID}" != "ubuntu" || "${VERSION_ID}" != "24.04" ]]; then
  echo "Detected ${PRETTY_NAME}."
  echo "This script targets Ubuntu 24.04 with ROS 2 Jazzy."
  exit 1
fi

ensure_bashrc_line() {
  local line="$1"
  grep -Fqx "${line}" "${HOME}/.bashrc" || echo "${line}" >> "${HOME}/.bashrc"
}

source_script_safely() {
  local script_path="$1"
  set +u
  # shellcheck disable=SC1090
  source "${script_path}"
  set -u
}

cleanup_stale_apt_sources() {
  sudo rm -f /etc/apt/sources.list.d/ros2.list
  sudo rm -f /etc/apt/sources.list.d/gazebo-stable.list
}

configure_ros_apt_repo() {
  local version=""
  local codename="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"
  local deb_path=""

  version="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F 'tag_name' | awk -F'"' '{print $4}')"
  deb_path="/tmp/ros2-apt-source_${version}_${codename}.deb"

  curl -fL -o "${deb_path}" \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${version}/ros2-apt-source_${version}.${codename}_all.deb"

  sudo rm -f /etc/apt/sources.list.d/ros2.list
  sudo dpkg -i "${deb_path}"
}

configure_gazebo_repo() {
  local codename="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"

  sudo mkdir -p /usr/share/keyrings
  sudo rm -f /etc/apt/sources.list.d/gazebo-stable.list
  sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable ${codename} main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
}

clone_or_update() {
  local branch="$1"
  local repo_url="$2"
  local dst="$3"

  if [[ ! -d "${dst}/.git" ]]; then
    git clone -b "${branch}" --depth 1 "${repo_url}" "${dst}"
    return
  fi

  git -C "${dst}" fetch --depth 1 origin "${branch}"
  git -C "${dst}" checkout "${branch}"
  git -C "${dst}" pull --ff-only origin "${branch}"
}

echo "Configuring locale and apt sources for ROS 2 Jazzy..."
cleanup_stale_apt_sources
sudo apt update
sudo apt install -y locales software-properties-common curl git gnupg lsb-release build-essential cmake
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo add-apt-repository universe -y

configure_ros_apt_repo

echo "Installing ROS 2 Jazzy, Nav2, and Gazebo Harmonic..."
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
  ros-jazzy-desktop \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-jazzy-cartographer \
  ros-jazzy-cartographer-ros \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup

configure_gazebo_repo
sudo apt update
sudo apt install -y gz-harmonic

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update

mkdir -p "${SRC_DIR}"

echo "Syncing TurtleBot3 sources into ${SRC_DIR}..."
clone_or_update jazzy https://github.com/ROBOTIS-GIT/DynamixelSDK.git "${SRC_DIR}/DynamixelSDK"
clone_or_update jazzy https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git "${SRC_DIR}/turtlebot3_msgs"
clone_or_update jazzy https://github.com/ROBOTIS-GIT/turtlebot3.git "${SRC_DIR}/turtlebot3"
clone_or_update jazzy https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git "${SRC_DIR}/turtlebot3_simulations"

rm -rf "${TASK7_DST}"
cp -R "${TASK7_SRC}" "${TASK7_DST}"

source_script_safely "${ROS_SETUP}"
cd "${WS_DIR}"

echo "Installing remaining rosdep dependencies..."
rosdep install --from-paths src --ignore-src -r -y

echo "Building the ROS 2 workspace..."
colcon build --symlink-install

ensure_bashrc_line "source ${ROS_SETUP}"
ensure_bashrc_line "source ${WS_DIR}/install/setup.bash"
ensure_bashrc_line "export TURTLEBOT3_MODEL=burger"
ensure_bashrc_line "export ROS_DOMAIN_ID=30 #TURTLEBOT3"

echo
echo "Task 7 setup complete."
echo "Workspace: ${WS_DIR}"
echo "Next step: ${SCRIPT_DIR}/run_task7.sh"
