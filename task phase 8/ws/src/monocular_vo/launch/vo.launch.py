from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("monocular_vo"))
    default_config = package_share / "config" / "vo.yaml"

    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=str(default_config),
        description="Path to the monocular_vo parameter file.",
    )

    video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value="/project/video.mp4",
        description="Absolute path to the replay video inside the container.",
    )

    camera_x_arg = DeclareLaunchArgument(
        "camera_x",
        default_value="0.0",
        description="base_link to camera_link translation X in meters.",
    )

    camera_y_arg = DeclareLaunchArgument(
        "camera_y",
        default_value="0.0",
        description="base_link to camera_link translation Y in meters.",
    )

    camera_z_arg = DeclareLaunchArgument(
        "camera_z",
        default_value="0.0",
        description="base_link to camera_link translation Z in meters.",
    )

    video_bridge = Node(
        package="monocular_vo",
        executable="video_bridge",
        name="video_bridge",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            {"video_path": LaunchConfiguration("video_path")},
        ],
    )

    vo_node = Node(
        package="monocular_vo",
        executable="vo_node",
        name="vo_node",
        output="screen",
        parameters=[LaunchConfiguration("config_file")],
    )

    base_to_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_camera_link_tf",
        arguments=[
            LaunchConfiguration("camera_x"),
            LaunchConfiguration("camera_y"),
            LaunchConfiguration("camera_z"),
            "0", "0", "0", "1",
            "base_link", "camera_link",
        ],
    )

    camera_to_optical = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_link_to_optical_tf",
        arguments=[
            "0", "0", "0",
            "-0.5", "0.5", "-0.5", "0.5",
            "camera_link", "camera_optical_frame",
        ],
    )

    return LaunchDescription(
        [
            config_arg,
            video_path_arg,
            camera_x_arg,
            camera_y_arg,
            camera_z_arg,
            base_to_camera,
            camera_to_optical,
            video_bridge,
            vo_node,
        ]
    )
