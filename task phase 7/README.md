# Task 7: TurtleBot3 Burger Nav2 + RViz Demo

This Task 7 deliverable targets `Ubuntu 24.04` with `ROS 2 Jazzy` and follows the official TurtleBot3 Jazzy quick-start and navigation simulation flow.

Official references:

- [ROS 2 Jazzy Ubuntu 24.04 install](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- [TurtleBot3 Quick Start (Jazzy)](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)
- [TurtleBot3 Navigation Simulation](https://emanual.robotis.com/docs/en/platform/turtlebot3/nav_simulation/)

## What Is Included

- `setup_linux_vm.sh`: installs ROS 2 Jazzy, Gazebo Harmonic, Nav2, TurtleBot3 sources, and this Task 7 package into `~/turtlebot3_ws`
- `run_task7.sh`: launches Gazebo in the background, then opens Nav2 + RViz in the foreground
- `task7_nav2_demo/`: lightweight ROS 2 package with launch files, stock map, RViz config, and fixed snack poses

## Workspace Layout

- `task7_nav2_demo/launch/gazebo_world.launch.py`
- `task7_nav2_demo/launch/nav2_rviz.launch.py`
- `task7_nav2_demo/maps/map.yaml`
- `task7_nav2_demo/maps/map.pgm`
- `task7_nav2_demo/rviz/tb3_navigation2.rviz`
- `task7_nav2_demo/config/snack_locations.yaml`

## Setup

Run the setup script once on Ubuntu 24.04:

```bash
cd /absolute/path/to/task\ phase\ 7
chmod +x setup_linux_vm.sh run_task7.sh
./setup_linux_vm.sh
```

What the setup script does:

- installs ROS 2 Jazzy from the official ROS apt source
- installs Gazebo Harmonic and Nav2 dependencies
- clones Jazzy branches of `DynamixelSDK`, `turtlebot3_msgs`, `turtlebot3`, and `turtlebot3_simulations`
- copies `task7_nav2_demo` into `~/turtlebot3_ws/src/`
- runs `rosdep install` and `colcon build --symlink-install`

## Run Task 7

Use the wrapper script:

```bash
cd /absolute/path/to/task\ phase\ 7
./run_task7.sh
```

Manual equivalents:

```bash
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

ros2 launch task7_nav2_demo gazebo_world.launch.py
ros2 launch task7_nav2_demo nav2_rviz.launch.py
ros2 run turtlebot3_teleop teleop_keyboard
```

## Operator Flow In RViz

1. Wait until Gazebo, Nav2, and RViz are fully up.
2. In RViz, click `2D Pose Estimate`.
3. Set the initial pose near the Burger spawn point around `x=-2.0`, `y=-0.5`, facing roughly toward the map center.
4. Keep refining the pose until the laser scan overlays the map cleanly.
5. If needed, run `ros2 run turtlebot3_teleop teleop_keyboard`, move the robot gently, then stop teleop with `Ctrl+C`.
6. Click `Navigation2 Goal` in RViz.
7. Send the three snack goals one by one using the coordinates from `task7_nav2_demo/config/snack_locations.yaml`.

## Snack Goals

These three fixed map-frame goals are the single source of truth for Task 7:

| Name | x | y | yaw |
| --- | ---: | ---: | ---: |
| `snack_left` | `-1.80` | `0.90` | `0.00` |
| `snack_top` | `0.90` | `1.80` | `-1.57` |
| `snack_right` | `1.80` | `-0.90` | `3.14` |

## Validation Checklist

- RViz shows `/map`, laser scan, TF, and Nav2 overlays.
- `ros2 node list` includes `amcl`, `controller_server`, `planner_server`, `bt_navigator`, and `map_server`.
- The Burger accepts goals from RViz and moves autonomously.
- Each snack goal succeeds before sending the next one.
- The robot avoids obstacles instead of driving through them.

## Invalid Goal Test

To verify safe failure, send a goal into an occupied or unknown area:

- click directly on the center obstacle around `x=0.0`, `y=0.0`, or
- click in the gray unknown space outside the white explored map region

Nav2 should reject or fail the goal without uncontrolled motion.

## Notes On Ubuntu 24.04

- The setup script now targets the same workspace pattern used in the official TurtleBot3 Jazzy docs: `~/turtlebot3_ws`.
- ROS 2 repository setup now uses the official `ros2-apt-source` package flow directly and removes any stale hand-written `ros2.list` file before reinstalling the source package.
- Gazebo Harmonic is configured with the OSRF apt repository exactly as described in the TurtleBot3 Jazzy guide.

## If You Already Hit The `sources could not be read` Error

An earlier run may have left a broken apt source file behind. Clean it once, then rerun setup:

```bash
sudo rm -f /etc/apt/sources.list.d/ros2.list
sudo rm -f /etc/apt/sources.list.d/gazebo-stable.list
sudo apt clean
sudo apt update
cd /absolute/path/to/task\ phase\ 7
./setup_linux_vm.sh
```
