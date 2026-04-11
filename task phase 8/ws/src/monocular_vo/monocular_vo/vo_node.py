from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from monocular_vo.frontend import TrackingConfig, detect_features, track_features
from monocular_vo.geometry import estimate_relative_pose
from monocular_vo.io import CameraProcessingState, build_camera_processing_state, preprocess_frame
from monocular_vo.pose_integration import (
    PlanarPose,
    base_to_camera_optical_transform,
    initial_world_from_camera_optical,
    integrate_camera_motion,
    planar_pose_from_world_transform,
    quaternion_from_yaw,
    world_from_base_transform,
)


def diagonal_covariance(values: list[float]) -> list[float]:
    covariance = [0.0] * 36
    for index, value in enumerate(values):
        covariance[index * 6 + index] = float(value)
    return covariance


@dataclass(frozen=True)
class DebugTrackOverlay:
    previous_points: np.ndarray
    current_points: np.ndarray
    rejected_previous_points: np.ndarray
    rejected_current_points: np.ndarray
    status: str
    tracked_count: int
    inlier_count: int
    reseeded: bool


class MonocularVoNode(Node):
    def __init__(self) -> None:
        super().__init__("vo_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("odom_topic", "/vo/odom")
        self.declare_parameter("path_topic", "/vo/path")
        self.declare_parameter("debug_image_topic", "/vo/debug_tracks")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("camera_link_frame_id", "camera_link")
        self.declare_parameter("camera_optical_frame_id", "camera_optical_frame")
        self.declare_parameter("use_rectified_images", False)
        self.declare_parameter("enable_clahe", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("camera_height_m", 1.3)
        self.declare_parameter("max_features", 1500)
        self.declare_parameter("min_inliers", 60)
        self.declare_parameter("min_tracked_points", 120)
        self.declare_parameter("min_feature_distance_px", 10.0)
        self.declare_parameter("shi_tomasi_quality_level", 0.01)
        self.declare_parameter("shi_tomasi_block_size", 7)
        self.declare_parameter("lk_win_size", 21)
        self.declare_parameter("lk_max_level", 3)
        self.declare_parameter("lk_max_iterations", 30)
        self.declare_parameter("lk_epsilon", 0.01)
        self.declare_parameter("ransac_threshold_px", 1.0)
        self.declare_parameter("translation_step_scale", 1.0)
        self.declare_parameter("feature_border_px", 20)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        self.debug_image_topic = self.get_parameter("debug_image_topic").get_parameter_value().string_value
        self.odom_frame_id = self.get_parameter("odom_frame_id").get_parameter_value().string_value
        self.base_frame_id = self.get_parameter("base_frame_id").get_parameter_value().string_value
        self.use_rectified_images = bool(self.get_parameter("use_rectified_images").value)
        self.enable_clahe = bool(self.get_parameter("enable_clahe").value)
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.camera_height_m = float(self.get_parameter("camera_height_m").value)
        self.min_inliers = int(self.get_parameter("min_inliers").value)
        self.ransac_threshold_px = float(self.get_parameter("ransac_threshold_px").value)
        self.translation_step_scale = float(self.get_parameter("translation_step_scale").value)

        self.tracking_config = TrackingConfig(
            max_features=int(self.get_parameter("max_features").value),
            min_tracked_points=int(self.get_parameter("min_tracked_points").value),
            min_feature_distance_px=float(self.get_parameter("min_feature_distance_px").value),
            shi_tomasi_quality_level=float(self.get_parameter("shi_tomasi_quality_level").value),
            shi_tomasi_block_size=int(self.get_parameter("shi_tomasi_block_size").value),
            lk_win_size=int(self.get_parameter("lk_win_size").value),
            lk_max_level=int(self.get_parameter("lk_max_level").value),
            lk_max_iterations=int(self.get_parameter("lk_max_iterations").value),
            lk_epsilon=float(self.get_parameter("lk_epsilon").value),
            feature_border_px=int(self.get_parameter("feature_border_px").value),
        )

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.processing_state: Optional[CameraProcessingState] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame_id
        self.path_msg.poses = []
        self.frame_counter = 0
        self.missing_camera_info_warned = False
        self.previous_gray: Optional[np.ndarray] = None
        self.previous_points = np.empty((0, 2), dtype=np.float32)
        self.world_from_camera_optical = initial_world_from_camera_optical()
        self.base_to_camera_optical = base_to_camera_optical_transform()
        self.last_planar_pose = PlanarPose(x=0.0, y=0.0, yaw=0.0)
        self.last_status = "waiting"
        self.last_rejection_reason = ""
        self.last_track_overlay = DebugTrackOverlay(
            previous_points=np.empty((0, 2), dtype=np.float32),
            current_points=np.empty((0, 2), dtype=np.float32),
            rejected_previous_points=np.empty((0, 2), dtype=np.float32),
            rejected_current_points=np.empty((0, 2), dtype=np.float32),
            status="waiting",
            tracked_count=0,
            inlier_count=0,
            reseeded=False,
        )

        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self._on_image, qos_profile_sensor_data)

        self.get_logger().info(
            (
                f"Phase 1 VO node ready. Waiting on {self.image_topic} and {self.camera_info_topic}. "
                f"camera_height_m={self.camera_height_m:.2f}, use_rectified_images={self.use_rectified_images}, "
                f"enable_clahe={self.enable_clahe}, max_features={self.tracking_config.max_features}, "
                f"min_inliers={self.min_inliers}"
            )
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg
        if self.processing_state is None:
            self.processing_state = build_camera_processing_state(msg)
            distortion_enabled = self.processing_state.map1 is not None and self.processing_state.map2 is not None
            self.get_logger().info(
                (
                    f"CameraInfo received: {msg.width}x{msg.height}, "
                    f"distortion_correction={'on' if distortion_enabled and not self.use_rectified_images else 'off'}"
                )
            )

    def _on_image(self, msg: Image) -> None:
        if self.latest_camera_info is None or self.processing_state is None:
            if not self.missing_camera_info_warned:
                self.get_logger().warning("Skipping image because CameraInfo has not been initialized yet.")
                self.missing_camera_info_warned = True
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        grayscale, display_bgr = preprocess_frame(
            image_bgr=frame,
            processing_state=self.processing_state,
            use_rectified_images=self.use_rectified_images,
            enable_clahe=self.enable_clahe,
        )

        if self.previous_gray is None:
            self._initialize_tracking(grayscale, msg)
        else:
            self._process_tracking_frame(grayscale, msg)

        self._publish_pose_outputs(msg)
        if self.publish_debug_image:
            self._publish_debug_image(msg, grayscale, display_bgr)

        self.frame_counter += 1
        self.missing_camera_info_warned = False

    def _initialize_tracking(self, grayscale: np.ndarray, image_msg: Image) -> None:
        self.previous_points = detect_features(grayscale, self.tracking_config)
        self.previous_gray = grayscale
        self.world_from_camera_optical = initial_world_from_camera_optical()
        self.last_planar_pose = PlanarPose(x=0.0, y=0.0, yaw=0.0)
        if not self.path_msg.poses:
            self.path_msg.poses.append(self._pose_stamped_from_planar_pose(image_msg, self.last_planar_pose))
        self.last_status = "initialized"
        self.last_rejection_reason = ""
        self.last_track_overlay = DebugTrackOverlay(
            previous_points=np.empty((0, 2), dtype=np.float32),
            current_points=np.empty((0, 2), dtype=np.float32),
            rejected_previous_points=np.empty((0, 2), dtype=np.float32),
            rejected_current_points=np.empty((0, 2), dtype=np.float32),
            status="initialized",
            tracked_count=int(self.previous_points.shape[0]),
            inlier_count=0,
            reseeded=False,
        )

    def _process_tracking_frame(self, grayscale: np.ndarray, image_msg: Image) -> None:
        track_result = track_features(
            previous_gray=self.previous_gray,
            current_gray=grayscale,
            previous_points=self.previous_points,
            config=self.tracking_config,
        )
        reseeded = False

        if track_result.previous_points.shape[0] < 8:
            self.previous_points = detect_features(grayscale, self.tracking_config)
            self.previous_gray = grayscale
            self.last_status = "rejected"
            self.last_rejection_reason = "too_few_tracks"
            self.last_track_overlay = DebugTrackOverlay(
                previous_points=track_result.previous_points,
                current_points=track_result.current_points,
                rejected_previous_points=track_result.rejected_previous_points,
                rejected_current_points=track_result.rejected_current_points,
                status="rejected: too_few_tracks",
                tracked_count=int(track_result.previous_points.shape[0]),
                inlier_count=0,
                reseeded=True,
            )
            return

        relative_pose = estimate_relative_pose(
            previous_points=track_result.previous_points,
            current_points=track_result.current_points,
            camera_matrix=self.processing_state.camera_matrix,
            ransac_threshold_px=self.ransac_threshold_px,
            translation_step_scale=self.translation_step_scale,
        )
        if relative_pose is None:
            self.previous_points = detect_features(grayscale, self.tracking_config)
            self.previous_gray = grayscale
            self.last_status = "rejected"
            self.last_rejection_reason = "essential_failed"
            self.last_track_overlay = DebugTrackOverlay(
                previous_points=track_result.previous_points,
                current_points=track_result.current_points,
                rejected_previous_points=track_result.rejected_previous_points,
                rejected_current_points=track_result.rejected_current_points,
                status="rejected: geometry_failed",
                tracked_count=int(track_result.previous_points.shape[0]),
                inlier_count=0,
                reseeded=True,
            )
            return

        inlier_previous_points = track_result.previous_points[relative_pose.inlier_mask]
        inlier_current_points = track_result.current_points[relative_pose.inlier_mask]
        outlier_previous_points = track_result.previous_points[~relative_pose.inlier_mask]
        outlier_current_points = track_result.current_points[~relative_pose.inlier_mask]
        rejected_previous_points = np.vstack(
            [track_result.rejected_previous_points, outlier_previous_points]
        ) if track_result.rejected_previous_points.size or outlier_previous_points.size else np.empty((0, 2), dtype=np.float32)
        rejected_current_points = np.vstack(
            [track_result.rejected_current_points, outlier_current_points]
        ) if track_result.rejected_current_points.size or outlier_current_points.size else np.empty((0, 2), dtype=np.float32)

        if relative_pose.num_inliers < self.min_inliers:
            self.previous_points = detect_features(grayscale, self.tracking_config)
            self.previous_gray = grayscale
            self.last_status = "rejected"
            self.last_rejection_reason = "too_few_inliers"
            self.last_track_overlay = DebugTrackOverlay(
                previous_points=inlier_previous_points,
                current_points=inlier_current_points,
                rejected_previous_points=rejected_previous_points,
                rejected_current_points=rejected_current_points,
                status="rejected: too_few_inliers",
                tracked_count=int(track_result.previous_points.shape[0]),
                inlier_count=int(relative_pose.num_inliers),
                reseeded=True,
            )
            return

        self.world_from_camera_optical = integrate_camera_motion(
            world_from_camera_optical=self.world_from_camera_optical,
            current_from_previous=relative_pose.current_from_previous,
        )
        world_from_base = world_from_base_transform(
            world_from_camera_optical=self.world_from_camera_optical,
            base_to_camera_optical=self.base_to_camera_optical,
        )
        self.last_planar_pose = planar_pose_from_world_transform(world_from_base)

        accepted_pose = self._pose_stamped_from_planar_pose(image_msg, self.last_planar_pose)
        self.path_msg.poses.append(accepted_pose)

        self.previous_gray = grayscale
        self.previous_points = inlier_current_points
        if self.previous_points.shape[0] < self.tracking_config.min_tracked_points:
            self.previous_points = detect_features(grayscale, self.tracking_config)
            reseeded = True

        self.last_status = "accepted"
        self.last_rejection_reason = ""
        self.last_track_overlay = DebugTrackOverlay(
            previous_points=inlier_previous_points,
            current_points=inlier_current_points,
            rejected_previous_points=rejected_previous_points,
            rejected_current_points=rejected_current_points,
            status="accepted" if not reseeded else "accepted: reseeded",
            tracked_count=int(track_result.previous_points.shape[0]),
            inlier_count=int(relative_pose.num_inliers),
            reseeded=reseeded,
        )

    def _pose_stamped_from_planar_pose(self, image_msg: Image, planar_pose: PlanarPose) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = image_msg.header.stamp
        pose.header.frame_id = self.odom_frame_id
        pose.pose.position.x = planar_pose.x
        pose.pose.position.y = planar_pose.y
        pose.pose.position.z = 0.0
        quat_x, quat_y, quat_z, quat_w = quaternion_from_yaw(planar_pose.yaw)
        pose.pose.orientation.x = quat_x
        pose.pose.orientation.y = quat_y
        pose.pose.orientation.z = quat_z
        pose.pose.orientation.w = quat_w
        return pose

    def _publish_pose_outputs(self, image_msg: Image) -> None:
        odom_msg = Odometry()
        odom_msg.header.stamp = image_msg.header.stamp
        odom_msg.header.frame_id = self.odom_frame_id
        odom_msg.child_frame_id = self.base_frame_id
        odom_msg.pose.pose.position.x = self.last_planar_pose.x
        odom_msg.pose.pose.position.y = self.last_planar_pose.y
        quat_x, quat_y, quat_z, quat_w = quaternion_from_yaw(self.last_planar_pose.yaw)
        odom_msg.pose.pose.orientation.x = quat_x
        odom_msg.pose.pose.orientation.y = quat_y
        odom_msg.pose.pose.orientation.z = quat_z
        odom_msg.pose.pose.orientation.w = quat_w
        odom_msg.pose.covariance = diagonal_covariance([4.0, 4.0, 100.0, 100.0, 100.0, 10.0])
        odom_msg.twist.covariance = diagonal_covariance([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        self.odom_pub.publish(odom_msg)

        self.path_msg.header.stamp = image_msg.header.stamp
        self.path_pub.publish(self.path_msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = image_msg.header.stamp
        tf_msg.header.frame_id = self.odom_frame_id
        tf_msg.child_frame_id = self.base_frame_id
        tf_msg.transform.translation.x = self.last_planar_pose.x
        tf_msg.transform.translation.y = self.last_planar_pose.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = quat_x
        tf_msg.transform.rotation.y = quat_y
        tf_msg.transform.rotation.z = quat_z
        tf_msg.transform.rotation.w = quat_w
        self.tf_broadcaster.sendTransform(tf_msg)

    def _publish_debug_image(self, image_msg: Image, grayscale: np.ndarray, display_bgr: np.ndarray) -> None:
        debug_frame = display_bgr.copy()
        max_tracks_to_draw = 300

        rejected_count = min(int(self.last_track_overlay.rejected_previous_points.shape[0]), max_tracks_to_draw)
        for index in range(rejected_count):
            previous_point = self.last_track_overlay.rejected_previous_points[index]
            current_point = self.last_track_overlay.rejected_current_points[index]
            start = tuple(np.round(previous_point).astype(int))
            end = tuple(np.round(current_point).astype(int))
            cv2.line(debug_frame, start, end, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(debug_frame, end, 2, (0, 0, 255), -1, cv2.LINE_AA)

        accepted_count = min(int(self.last_track_overlay.previous_points.shape[0]), max_tracks_to_draw)
        for index in range(accepted_count):
            previous_point = self.last_track_overlay.previous_points[index]
            current_point = self.last_track_overlay.current_points[index]
            start = tuple(np.round(previous_point).astype(int))
            end = tuple(np.round(current_point).astype(int))
            cv2.line(debug_frame, start, end, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(debug_frame, end, 2, (0, 255, 0), -1, cv2.LINE_AA)

        lines = [
            "Task 8 Monocular VO Phase 1",
            f"Frame {self.frame_counter}",
            f"Image {display_bgr.shape[1]}x{display_bgr.shape[0]}",
            f"Status {self.last_track_overlay.status}",
            f"Tracks {self.last_track_overlay.tracked_count}  Inliers {self.last_track_overlay.inlier_count}",
            f"Pose x={self.last_planar_pose.x:.2f} y={self.last_planar_pose.y:.2f} yaw={np.degrees(self.last_planar_pose.yaw):.1f} deg",
            f"CLAHE {'on' if self.enable_clahe else 'off'}  Undistort {'off' if self.use_rectified_images else 'auto'}",
        ]

        for index, text in enumerate(lines):
            cv2.putText(
                debug_frame,
                text,
                (20, 40 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
        debug_msg.header.stamp = image_msg.header.stamp
        debug_msg.header.frame_id = image_msg.header.frame_id
        self.debug_pub.publish(debug_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MonocularVoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
