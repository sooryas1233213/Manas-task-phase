#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    turtlebot3_model = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_path = LaunchConfiguration(
        'map',
        default=os.path.join(
            get_package_share_directory('task7_nav2_demo'),
            'maps',
            'map.yaml',
        ),
    )
    params_file = LaunchConfiguration(
        'params_file',
        default=os.path.join(
            get_package_share_directory('turtlebot3_navigation2'),
            'param',
            f'{turtlebot3_model}.yaml',
        ),
    )
    rviz_config = LaunchConfiguration(
        'rviz_config',
        default=os.path.join(
            get_package_share_directory('task7_nav2_demo'),
            'rviz',
            'tb3_navigation2.rviz',
        ),
    )

    nav2_launch = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'bringup_launch.py',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=map_path,
            description='Full path to the occupancy-grid YAML file.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=params_file,
            description='Full path to the TurtleBot3 Nav2 parameter file.',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=rviz_config,
            description='Full path to the RViz config used for Task 7.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo simulation clock.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={
                'map': map_path,
                'params_file': params_file,
                'use_sim_time': use_sim_time,
            }.items(),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
    ])
