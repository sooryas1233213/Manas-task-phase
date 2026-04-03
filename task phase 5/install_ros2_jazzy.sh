#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME}") main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
sudo apt update
sudo apt install -y ros-jazzy-desktop

if ! grep -Fq "source /opt/ros/jazzy/setup.bash" "${HOME}/.bashrc"; then
  echo "source /opt/ros/jazzy/setup.bash" >> "${HOME}/.bashrc"
fi

source /opt/ros/jazzy/setup.bash
echo "ROS 2 Jazzy installation complete."
