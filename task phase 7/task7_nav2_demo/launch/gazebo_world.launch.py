#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    world_launch = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'launch',
        'turtlebot3_world.launch.py',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo simulation clock.',
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='-2.0',
            description='Initial TurtleBot3 Burger x pose in meters.',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='-0.5',
            description='Initial TurtleBot3 Burger y pose in meters.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(world_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'x_pose': x_pose,
                'y_pose': y_pose,
            }.items(),
        ),
    ])
