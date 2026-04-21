from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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

from monocular_vo.frontend import (
    TrackingConfig,
    TrackResult,
    detect_features,
    detect_orb_features,
    track_features,
)
from monocular_vo.geometry import RelativePoseEstimate, estimate_relative_pose, evaluate_homography_support
from monocular_vo.io import CameraProcessingState, build_camera_processing_state, preprocess_frame
from monocular_vo.local_map import (
    KeyframeRecord,
    LandmarkRecord,
    LocalMapConfig,
    PoseRefinementResult,
    collect_visible_landmarks,
    insert_keyframe,
    refine_current_pose,
)
from monocular_vo.pose_integration import (
    PlanarPose,
    base_to_camera_optical_transform,
    current_base_from_previous_base_transform,
    initial_world_from_camera_optical,
    integrate_base_motion,
    integrate_camera_motion,
    integrate_camera_rotation_only,
    planar_step_length_from_relative_base_transform,
    planar_pose_from_world_transform,
    project_base_motion_to_planar,
    project_base_rotation_to_yaw,
    quaternion_from_yaw,
    rotation_matrix_from_yaw,
    scaled_camera_motion_transform,
    world_from_base_transform,
    world_from_planar_base_pose,
)
from monocular_vo.scale import ScaleConfig, ScaleEstimate, estimate_ground_plane_scale, hold_scale_estimate


def diagonal_covariance(values: list[float]) -> list[float]:
    covariance = [0.0] * 36
    for index, value in enumerate(values):
        covariance[index * 6 + index] = float(value)
    return covariance


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _merge_points(*point_sets: np.ndarray) -> np.ndarray:
    valid_sets = [points.reshape(-1, 2).astype(np.float32) for points in point_sets if points.size > 0]
    if not valid_sets:
        return _empty_points()
    return np.vstack(valid_sets)


@dataclass(frozen=True)
class MotionDecision:
    kind: str
    reason: str


@dataclass
class TrackState:
    track_id: int
    age: int
    observations: deque[np.ndarray]


@dataclass(frozen=True)
class DebugTrackOverlay:
    accepted_previous_points: np.ndarray
    accepted_current_points: np.ndarray
    rejected_previous_points: np.ndarray
    rejected_current_points: np.ndarray
    decision: str
    reason: str
    total_input_tracks: int
    forward_survivors: int
    fb_survivors: int
    inlier_count: int
    median_fb_error_px: float
    median_epipolar_error_px: float
    median_parallax_px: float
    homography_inliers: int
    rotation_angle_deg: float
    reseeded: bool
    scale_candidate_count: int
    triangulated_count: int
    plane_inlier_count: int
    plane_inlier_ratio: float
    raw_step_scale_m: float | None
    filtered_step_scale_m: float
    scale_confidence: float
    scale_status: str
    vehicle_projection_used: bool
    active_keyframes: int
    active_landmarks: int
    visible_landmark_count: int
    refinement_attempted: bool
    refinement_applied: bool
    refinement_rmse_before: float
    refinement_rmse_after: float
    refinement_status: str


def _default_debug_overlay() -> DebugTrackOverlay:
    return DebugTrackOverlay(
        accepted_previous_points=_empty_points(),
        accepted_current_points=_empty_points(),
        rejected_previous_points=_empty_points(),
        rejected_current_points=_empty_points(),
        decision="waiting",
        reason="",
        total_input_tracks=0,
        forward_survivors=0,
        fb_survivors=0,
        inlier_count=0,
        median_fb_error_px=float("inf"),
        median_epipolar_error_px=float("inf"),
        median_parallax_px=0.0,
        homography_inliers=0,
        rotation_angle_deg=0.0,
        reseeded=False,
        scale_candidate_count=0,
        triangulated_count=0,
        plane_inlier_count=0,
        plane_inlier_ratio=0.0,
        raw_step_scale_m=None,
        filtered_step_scale_m=1.0,
        scale_confidence=0.0,
        scale_status="waiting",
        vehicle_projection_used=False,
        active_keyframes=0,
        active_landmarks=0,
        visible_landmark_count=0,
        refinement_attempted=False,
        refinement_applied=False,
        refinement_rmse_before=float("inf"),
        refinement_rmse_after=float("inf"),
        refinement_status="waiting",
    )


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
        self.declare_parameter("camera_translation_x", 0.0)
        self.declare_parameter("camera_translation_y", 0.0)
        self.declare_parameter("camera_translation_z", 0.0)
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
        self.declare_parameter("klt_fb_max_error_px", 1.5)
        self.declare_parameter("min_geometry_points", 30)
        self.declare_parameter("min_parallax_px", 2.0)
        self.declare_parameter("max_median_epipolar_error_px", 1.5)
        self.declare_parameter("enable_homography_gate", True)
        self.declare_parameter("homography_dominance_ratio", 1.15)
        self.declare_parameter("homography_ransac_threshold_px", 3.0)
        self.declare_parameter("yaw_only_min_rotation_deg", 0.4)
        self.declare_parameter("orb_reseed_threshold", 100)
        self.declare_parameter("orb_max_features", 600)
        self.declare_parameter("grid_rows", 4)
        self.declare_parameter("grid_cols", 8)
        self.declare_parameter("consecutive_rejects_for_reseed", 2)
        self.declare_parameter("scale_mode", "ground_plane")
        self.declare_parameter("min_scale_track_age", 3)
        self.declare_parameter("ground_region_min_y_frac", 0.55)
        self.declare_parameter("triangulation_min_parallax_px", 3.0)
        self.declare_parameter("ground_flow_angle_tolerance_deg", 20.0)
        self.declare_parameter("min_scale_candidate_points", 30)
        self.declare_parameter("min_plane_inliers", 20)
        self.declare_parameter("min_plane_inlier_ratio", 0.5)
        self.declare_parameter("max_ground_normal_deviation_deg", 35.0)
        self.declare_parameter("scale_ema_alpha", 0.25)
        self.declare_parameter("max_scale_jump_ratio", 1.5)
        self.declare_parameter("min_scale_confidence", 0.55)
        self.declare_parameter("min_step_scale_m", 0.01)
        self.declare_parameter("max_step_scale_m", 5.0)
        self.declare_parameter("enable_vehicle_motion_projection", True)
        self.declare_parameter("enable_local_map_refinement", True)
        self.declare_parameter("keyframe_min_accepted_frames", 8)
        self.declare_parameter("keyframe_force_frames", 12)
        self.declare_parameter("keyframe_rotation_thresh_deg", 4.0)
        self.declare_parameter("keyframe_parallax_thresh_px", 18.0)
        self.declare_parameter("keyframe_track_overlap_ratio", 0.70)
        self.declare_parameter("local_map_max_keyframes", 3)
        self.declare_parameter("local_map_min_landmarks", 40)
        self.declare_parameter("local_refine_max_iterations", 8)
        self.declare_parameter("local_refine_max_reprojection_error_px", 2.0)
        self.declare_parameter("local_refine_max_yaw_update_deg", 2.0)
        self.declare_parameter("local_refine_max_translation_update_ratio", 0.15)
        self.declare_parameter("diagnostics_csv_path", "")

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
        self.camera_translation_xyz = (
            float(self.get_parameter("camera_translation_x").value),
            float(self.get_parameter("camera_translation_y").value),
            float(self.get_parameter("camera_translation_z").value),
        )
        self.min_inliers = int(self.get_parameter("min_inliers").value)
        self.ransac_threshold_px = float(self.get_parameter("ransac_threshold_px").value)
        self.translation_step_scale = float(self.get_parameter("translation_step_scale").value)
        self.min_geometry_points = int(self.get_parameter("min_geometry_points").value)
        self.min_parallax_px = float(self.get_parameter("min_parallax_px").value)
        self.max_median_epipolar_error_px = float(self.get_parameter("max_median_epipolar_error_px").value)
        self.enable_homography_gate = bool(self.get_parameter("enable_homography_gate").value)
        self.homography_dominance_ratio = float(self.get_parameter("homography_dominance_ratio").value)
        self.homography_ransac_threshold_px = float(self.get_parameter("homography_ransac_threshold_px").value)
        self.yaw_only_min_rotation_deg = float(self.get_parameter("yaw_only_min_rotation_deg").value)
        self.orb_reseed_threshold = int(self.get_parameter("orb_reseed_threshold").value)
        self.consecutive_rejects_for_reseed = int(self.get_parameter("consecutive_rejects_for_reseed").value)
        self.enable_vehicle_motion_projection = bool(self.get_parameter("enable_vehicle_motion_projection").value)
        self.enable_local_map_refinement = bool(self.get_parameter("enable_local_map_refinement").value)
        self.diagnostics_csv_path = self.get_parameter("diagnostics_csv_path").get_parameter_value().string_value
        self.scale_config = ScaleConfig(
            scale_mode=self.get_parameter("scale_mode").get_parameter_value().string_value,
            min_scale_track_age=int(self.get_parameter("min_scale_track_age").value),
            ground_region_min_y_frac=float(self.get_parameter("ground_region_min_y_frac").value),
            triangulation_min_parallax_px=float(self.get_parameter("triangulation_min_parallax_px").value),
            ground_flow_angle_tolerance_deg=float(self.get_parameter("ground_flow_angle_tolerance_deg").value),
            min_scale_candidate_points=int(self.get_parameter("min_scale_candidate_points").value),
            min_plane_inliers=int(self.get_parameter("min_plane_inliers").value),
            min_plane_inlier_ratio=float(self.get_parameter("min_plane_inlier_ratio").value),
            max_ground_normal_deviation_deg=float(self.get_parameter("max_ground_normal_deviation_deg").value),
            scale_ema_alpha=float(self.get_parameter("scale_ema_alpha").value),
            max_scale_jump_ratio=float(self.get_parameter("max_scale_jump_ratio").value),
            min_scale_confidence=float(self.get_parameter("min_scale_confidence").value),
            min_step_scale_m=float(self.get_parameter("min_step_scale_m").value),
            max_step_scale_m=float(self.get_parameter("max_step_scale_m").value),
            bootstrap_scale_m=self.translation_step_scale,
        )
        self.local_map_config = LocalMapConfig(
            max_keyframes=int(self.get_parameter("local_map_max_keyframes").value),
            keyframe_min_accepted_frames=int(self.get_parameter("keyframe_min_accepted_frames").value),
            keyframe_force_frames=int(self.get_parameter("keyframe_force_frames").value),
            keyframe_rotation_thresh_deg=float(self.get_parameter("keyframe_rotation_thresh_deg").value),
            keyframe_parallax_thresh_px=float(self.get_parameter("keyframe_parallax_thresh_px").value),
            keyframe_track_overlap_ratio=float(self.get_parameter("keyframe_track_overlap_ratio").value),
            min_landmarks=int(self.get_parameter("local_map_min_landmarks").value),
            max_reprojection_error_px=float(self.get_parameter("local_refine_max_reprojection_error_px").value),
            max_iterations=int(self.get_parameter("local_refine_max_iterations").value),
            max_yaw_update_deg=float(self.get_parameter("local_refine_max_yaw_update_deg").value),
            max_translation_update_ratio=float(self.get_parameter("local_refine_max_translation_update_ratio").value),
        )

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
            klt_fb_max_error_px=float(self.get_parameter("klt_fb_max_error_px").value),
            orb_max_features=int(self.get_parameter("orb_max_features").value),
            grid_rows=int(self.get_parameter("grid_rows").value),
            grid_cols=int(self.get_parameter("grid_cols").value),
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
        self.previous_points = _empty_points()
        self.previous_track_ids = np.empty((0,), dtype=np.int64)
        self.track_states: dict[int, TrackState] = {}
        self.next_track_id = 0
        self.world_from_camera_optical = initial_world_from_camera_optical(self.camera_translation_xyz)
        self.base_to_camera_optical = base_to_camera_optical_transform(self.camera_translation_xyz)
        self.expected_ground_normal_camera = self._expected_ground_normal_camera()
        self.last_planar_pose = PlanarPose(x=0.0, y=0.0, yaw=0.0)
        self.consecutive_reject_count = 0
        self.last_track_overlay = _default_debug_overlay()
        self.last_stable_scale_m: float | None = None
        self.last_scale_estimate = hold_scale_estimate(
            last_stable_scale_m=None,
            bootstrap_scale_m=self.translation_step_scale,
            reason="bootstrap_scale_only",
        )
        self.keyframes: deque[KeyframeRecord] = deque()
        self.landmarks: dict[int, LandmarkRecord] = {}
        self.accepted_full_pose_count = 0
        self._initialize_diagnostics_file()

        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self._on_image, qos_profile_sensor_data)

        self.get_logger().info(
            (
                f"Phase 3 VO node ready. Waiting on {self.image_topic} and {self.camera_info_topic}. "
                f"max_features={self.tracking_config.max_features}, min_inliers={self.min_inliers}, "
                f"min_geometry_points={self.min_geometry_points}, orb_reseed_threshold={self.orb_reseed_threshold}, "
                f"scale_mode={self.scale_config.scale_mode}, "
                f"vehicle_projection={'on' if self.enable_vehicle_motion_projection else 'off'}, "
                f"local_refine={'on' if self.enable_local_map_refinement else 'off'}"
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
            self._publish_debug_image(msg, display_bgr)
        self._write_diagnostics_row(msg)

        self.frame_counter += 1
        self.missing_camera_info_warned = False

    def _initialize_tracking(self, grayscale: np.ndarray, image_msg: Image) -> None:
        self.previous_points = detect_features(grayscale, self.tracking_config)
        self.previous_track_ids = self._assign_new_track_ids(self.previous_points.shape[0])
        self.track_states = self._build_track_states(self.previous_track_ids, self.previous_points)
        self.previous_gray = grayscale
        self.world_from_camera_optical = initial_world_from_camera_optical(self.camera_translation_xyz)
        self.last_planar_pose = PlanarPose(x=0.0, y=0.0, yaw=0.0)
        self.consecutive_reject_count = 0
        self.last_stable_scale_m = None
        self.last_scale_estimate = hold_scale_estimate(
            last_stable_scale_m=None,
            bootstrap_scale_m=self.translation_step_scale,
            reason="bootstrap_scale_only",
        )
        self.keyframes.clear()
        self.landmarks.clear()
        self.accepted_full_pose_count = 0
        if not self.path_msg.poses:
            self.path_msg.poses.append(self._pose_stamped_from_planar_pose(image_msg, self.last_planar_pose))

        self.last_track_overlay = DebugTrackOverlay(
            accepted_previous_points=_empty_points(),
            accepted_current_points=_empty_points(),
            rejected_previous_points=_empty_points(),
            rejected_current_points=_empty_points(),
            decision="initialized",
            reason="seeded_features",
            total_input_tracks=int(self.previous_points.shape[0]),
            forward_survivors=0,
            fb_survivors=0,
            inlier_count=0,
            median_fb_error_px=float("inf"),
            median_epipolar_error_px=float("inf"),
            median_parallax_px=0.0,
            homography_inliers=0,
            rotation_angle_deg=0.0,
            reseeded=False,
            scale_candidate_count=0,
            triangulated_count=0,
            plane_inlier_count=0,
            plane_inlier_ratio=0.0,
            raw_step_scale_m=None,
            filtered_step_scale_m=self.last_scale_estimate.filtered_step_scale_m,
            scale_confidence=0.0,
            scale_status=self.last_scale_estimate.reason,
            vehicle_projection_used=False,
            active_keyframes=0,
            active_landmarks=0,
            visible_landmark_count=0,
            refinement_attempted=False,
            refinement_applied=False,
            refinement_rmse_before=float("inf"),
            refinement_rmse_after=float("inf"),
            refinement_status="seeded_features",
        )

    def _process_tracking_frame(self, grayscale: np.ndarray, image_msg: Image) -> None:
        track_result = track_features(
            previous_gray=self.previous_gray,
            current_gray=grayscale,
            previous_points=self.previous_points,
            config=self.tracking_config,
        )

        if track_result.health.fb_survivor_count < self.min_geometry_points:
            self._handle_reject_hold(
                grayscale=grayscale,
                track_result=track_result,
                relative_pose=None,
                reason="too_few_fb_tracks",
            )
            return

        relative_pose = estimate_relative_pose(
            previous_points=track_result.previous_points,
            current_points=track_result.current_points,
            camera_matrix=self.processing_state.camera_matrix,
            ransac_threshold_px=self.ransac_threshold_px,
        )
        if relative_pose is None:
            self._handle_reject_hold(
                grayscale=grayscale,
                track_result=track_result,
                relative_pose=None,
                reason="geometry_failed",
            )
            return

        survivor_track_ids = self.previous_track_ids[track_result.survivor_indices]
        self._maybe_apply_homography_gate(track_result, relative_pose)
        decision = self._decide_motion(relative_pose)

        if decision.kind == "accept_full_pose":
            self._handle_accept(
                grayscale=grayscale,
                image_msg=image_msg,
                track_result=track_result,
                relative_pose=relative_pose,
                survivor_track_ids=survivor_track_ids,
                decision=decision,
                yaw_only=False,
            )
            return

        if decision.kind == "accept_yaw_only":
            self._handle_accept(
                grayscale=grayscale,
                image_msg=image_msg,
                track_result=track_result,
                relative_pose=relative_pose,
                survivor_track_ids=survivor_track_ids,
                decision=decision,
                yaw_only=True,
            )
            return

        self._handle_reject_hold(
            grayscale=grayscale,
            track_result=track_result,
            relative_pose=relative_pose,
            reason=decision.reason,
        )

    def _expected_ground_normal_camera(self) -> np.ndarray:
        expected = self.base_to_camera_optical[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(expected))
        if norm <= 1e-9:
            return np.array([0.0, -1.0, 0.0], dtype=np.float64)
        return expected / norm

    def _assign_new_track_ids(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        start = self.next_track_id
        self.next_track_id += int(count)
        return np.arange(start, start + int(count), dtype=np.int64)

    def _build_track_states(
        self,
        track_ids: np.ndarray,
        points: np.ndarray,
        preserve_existing: bool = False,
    ) -> dict[int, TrackState]:
        active_states: dict[int, TrackState] = {}
        for track_id, point in zip(track_ids.tolist(), points.reshape(-1, 2)):
            existing_state = self.track_states.get(int(track_id)) if preserve_existing else None
            if existing_state is None:
                active_states[int(track_id)] = TrackState(
                    track_id=int(track_id),
                    age=1,
                    observations=deque([np.asarray(point, dtype=np.float32)], maxlen=4),
                )
            else:
                active_states[int(track_id)] = existing_state
        return active_states

    def _update_live_track_states(self, track_ids: np.ndarray, current_points: np.ndarray) -> None:
        updated_states: dict[int, TrackState] = {}
        for track_id, current_point in zip(track_ids.tolist(), current_points.reshape(-1, 2)):
            existing_state = self.track_states.get(int(track_id))
            if existing_state is None:
                observations: deque[np.ndarray] = deque(maxlen=4)
                age = 0
            else:
                observations = deque(existing_state.observations, maxlen=4)
                age = existing_state.age
            observations.append(np.asarray(current_point, dtype=np.float32))
            updated_states[int(track_id)] = TrackState(
                track_id=int(track_id),
                age=int(age) + 1,
                observations=observations,
            )
        self.track_states = updated_states

    def _estimate_scale_for_frame(
        self,
        previous_points: np.ndarray,
        current_points: np.ndarray,
        track_ids: np.ndarray,
        image_height: int,
        relative_pose: RelativePoseEstimate,
    ) -> ScaleEstimate:
        track_ages = np.array(
            [self.track_states.get(int(track_id), TrackState(int(track_id), 0, deque(maxlen=4))).age + 1 for track_id in track_ids],
            dtype=np.int32,
        )
        return estimate_ground_plane_scale(
            previous_points=previous_points,
            current_points=current_points,
            track_ages=track_ages,
            image_height=image_height,
            camera_matrix=self.processing_state.camera_matrix,
            rotation=relative_pose.rotation,
            translation_unit=relative_pose.translation,
            camera_height_m=self.camera_height_m,
            expected_up_vector=self.expected_ground_normal_camera,
            config=self.scale_config,
            last_stable_scale_m=self.last_stable_scale_m,
            rng=np.random.default_rng(self.frame_counter),
        )

    def _initialize_diagnostics_file(self) -> None:
        self._diagnostics_path: Path | None = None
        if not self.diagnostics_csv_path:
            return

        diagnostics_path = Path(self.diagnostics_csv_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = diagnostics_path.exists()
        self._diagnostics_path = diagnostics_path
        if file_exists:
            return

        with diagnostics_path.open("w", newline="") as diagnostics_file:
            writer = csv.writer(diagnostics_file)
            writer.writerow(
                [
                    "frame",
                    "stamp_sec",
                    "decision",
                    "reason",
                    "inliers",
                    "median_epipolar_error_px",
                    "median_parallax_px",
                    "homography_inliers",
                    "raw_step_scale_m",
                    "filtered_step_scale_m",
                    "vehicle_projection_used",
                    "keyframe_count",
                    "landmark_count",
                    "visible_landmark_count",
                    "refinement_attempted",
                    "refinement_applied",
                    "refinement_status",
                    "refinement_rmse_before",
                    "refinement_rmse_after",
                    "pose_x",
                    "pose_y",
                    "pose_yaw_deg",
                ]
            )

    def _write_diagnostics_row(self, image_msg: Image) -> None:
        if self._diagnostics_path is None:
            return

        stamp_sec = float(image_msg.header.stamp.sec) + float(image_msg.header.stamp.nanosec) * 1e-9
        with self._diagnostics_path.open("a", newline="") as diagnostics_file:
            writer = csv.writer(diagnostics_file)
            writer.writerow(
                [
                    self.frame_counter,
                    f"{stamp_sec:.9f}",
                    self.last_track_overlay.decision,
                    self.last_track_overlay.reason,
                    self.last_track_overlay.inlier_count,
                    self.last_track_overlay.median_epipolar_error_px,
                    self.last_track_overlay.median_parallax_px,
                    self.last_track_overlay.homography_inliers,
                    self.last_track_overlay.raw_step_scale_m,
                    self.last_track_overlay.filtered_step_scale_m,
                    int(self.last_track_overlay.vehicle_projection_used),
                    self.last_track_overlay.active_keyframes,
                    self.last_track_overlay.active_landmarks,
                    self.last_track_overlay.visible_landmark_count,
                    int(self.last_track_overlay.refinement_attempted),
                    int(self.last_track_overlay.refinement_applied),
                    self.last_track_overlay.refinement_status,
                    self.last_track_overlay.refinement_rmse_before,
                    self.last_track_overlay.refinement_rmse_after,
                    self.last_planar_pose.x,
                    self.last_planar_pose.y,
                    float(np.degrees(self.last_planar_pose.yaw)),
                ]
            )

    def _integrate_full_pose_step(
        self,
        relative_pose: RelativePoseEstimate,
        scale_estimate: ScaleEstimate,
    ) -> tuple[np.ndarray, np.ndarray, bool, float]:
        previous_world_from_base = world_from_base_transform(
            world_from_camera_optical=self.world_from_camera_optical,
            base_to_camera_optical=self.base_to_camera_optical,
        )
        scaled_current_from_previous = scaled_camera_motion_transform(
            rotation=relative_pose.rotation,
            translation_unit=relative_pose.translation,
            step_scale_m=scale_estimate.applied_step_scale_m,
        )
        current_from_previous_base = current_base_from_previous_base_transform(
            current_from_previous_camera=scaled_current_from_previous,
            base_to_camera_optical=self.base_to_camera_optical,
        )
        if self.enable_vehicle_motion_projection:
            projected_current_from_previous_base = project_base_motion_to_planar(current_from_previous_base)
            world_from_base = integrate_base_motion(
                world_from_base=previous_world_from_base,
                current_from_previous_base=projected_current_from_previous_base,
            )
            world_from_camera_optical = world_from_base @ self.base_to_camera_optical
            return (
                world_from_camera_optical,
                world_from_base,
                True,
                planar_step_length_from_relative_base_transform(projected_current_from_previous_base),
            )

        world_from_camera_optical = integrate_camera_motion(
            world_from_camera_optical=self.world_from_camera_optical,
            current_from_previous=scaled_current_from_previous,
        )
        world_from_base = world_from_base_transform(
            world_from_camera_optical=world_from_camera_optical,
            base_to_camera_optical=self.base_to_camera_optical,
        )
        return (
            world_from_camera_optical,
            world_from_base,
            False,
            planar_step_length_from_relative_base_transform(current_from_previous_base),
        )

    def _integrate_yaw_only_step(
        self,
        relative_pose: RelativePoseEstimate,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.enable_vehicle_motion_projection:
            previous_world_from_base = world_from_base_transform(
                world_from_camera_optical=self.world_from_camera_optical,
                base_to_camera_optical=self.base_to_camera_optical,
            )
            current_from_previous_base = current_base_from_previous_base_transform(
                current_from_previous_camera=relative_pose.current_from_previous,
                base_to_camera_optical=self.base_to_camera_optical,
            )
            projected_current_from_previous_base = project_base_rotation_to_yaw(current_from_previous_base)
            world_from_base = integrate_base_motion(
                world_from_base=previous_world_from_base,
                current_from_previous_base=projected_current_from_previous_base,
            )
            world_from_camera_optical = world_from_base @ self.base_to_camera_optical
            return world_from_camera_optical, world_from_base, True

        world_from_camera_optical = integrate_camera_rotation_only(
            world_from_camera_optical=self.world_from_camera_optical,
            current_from_previous=relative_pose.current_from_previous,
        )
        world_from_base = world_from_base_transform(
            world_from_camera_optical=world_from_camera_optical,
            base_to_camera_optical=self.base_to_camera_optical,
        )
        return world_from_camera_optical, world_from_base, False

    def _update_landmark_quality(self, refinement_result: PoseRefinementResult) -> None:
        if not refinement_result.accepted:
            return

        stale_track_ids: list[int] = []
        for track_id, residual_norm_px in zip(
            refinement_result.matched_track_ids.tolist(),
            refinement_result.residual_norms_px.tolist(),
        ):
            landmark = self.landmarks.get(int(track_id))
            if landmark is None:
                continue
            if np.isfinite(residual_norm_px) and residual_norm_px > self.local_map_config.max_reprojection_error_px * 2.0:
                stale_track_ids.append(int(track_id))
                continue
            self.landmarks[int(track_id)] = LandmarkRecord(
                track_id=landmark.track_id,
                world_point=np.asarray(landmark.world_point, dtype=np.float64),
                source_keyframe_indices=landmark.source_keyframe_indices,
                last_reprojection_error_px=float(residual_norm_px),
                support_count=int(landmark.support_count) + 1,
                last_seen_accepted_frame_index=self.accepted_full_pose_count,
            )

        for track_id in stale_track_ids:
            self.landmarks.pop(track_id, None)

    def _maybe_refine_current_pose(
        self,
        current_track_ids: np.ndarray,
        current_points: np.ndarray,
        image_shape: tuple[int, int],
        current_step_length_m: float,
    ) -> PoseRefinementResult:
        if not self.enable_local_map_refinement:
            return PoseRefinementResult(
                attempted=False,
                accepted=False,
                status="local_refine_disabled",
                refined_planar_pose=self.last_planar_pose,
                visible_landmark_count=0,
                grid_cell_count=0,
                rmse_before=float("inf"),
                rmse_after=float("inf"),
                matched_track_ids=np.empty((0,), dtype=np.int64),
                residual_norms_px=np.empty((0,), dtype=np.float64),
            )
        if not self.keyframes or not self.landmarks:
            return PoseRefinementResult(
                attempted=False,
                accepted=False,
                status="no_local_map_support",
                refined_planar_pose=self.last_planar_pose,
                visible_landmark_count=0,
                grid_cell_count=0,
                rmse_before=float("inf"),
                rmse_after=float("inf"),
                matched_track_ids=np.empty((0,), dtype=np.int64),
                residual_norms_px=np.empty((0,), dtype=np.float64),
            )

        matched_track_ids, world_points, image_points, grid_cell_count = collect_visible_landmarks(
            landmarks=self.landmarks,
            current_track_ids=current_track_ids,
            current_points=current_points,
            image_shape=image_shape,
            grid_rows=self.tracking_config.grid_rows,
            grid_cols=self.tracking_config.grid_cols,
        )
        refinement_result = refine_current_pose(
            current_planar_pose=self.last_planar_pose,
            matched_track_ids=matched_track_ids,
            world_points=world_points,
            image_points=image_points,
            grid_cell_count=grid_cell_count,
            camera_matrix=self.processing_state.camera_matrix,
            base_to_camera_optical=self.base_to_camera_optical,
            current_step_length_m=current_step_length_m,
            config=self.local_map_config,
        )
        if refinement_result.accepted:
            self.last_planar_pose = refinement_result.refined_planar_pose
            world_from_base = world_from_planar_base_pose(self.last_planar_pose)
            self.world_from_camera_optical = world_from_base @ self.base_to_camera_optical
            self._update_landmark_quality(refinement_result)
        return refinement_result

    def _maybe_insert_keyframe(
        self,
        live_track_ids: np.ndarray,
        live_points: np.ndarray,
        world_from_base: np.ndarray,
        rotation_angle_deg: float,
        median_parallax_px: float,
    ) -> int:
        if not self.enable_local_map_refinement or live_track_ids.size == 0 or live_points.size == 0:
            return 0

        track_points_by_id = {
            int(track_id): np.asarray(point, dtype=np.float32)
            for track_id, point in zip(live_track_ids.tolist(), live_points.reshape(-1, 2))
        }
        should_insert = False
        if not self.keyframes:
            should_insert = True
        else:
            latest_keyframe = self.keyframes[-1]
            accepted_frames_since_last_keyframe = self.accepted_full_pose_count - latest_keyframe.accepted_frame_index
            shared_track_count = len(set(track_points_by_id.keys()) & set(latest_keyframe.track_points_by_id.keys()))
            should_insert = accepted_frames_since_last_keyframe >= self.local_map_config.keyframe_force_frames
            if (
                not should_insert
                and accepted_frames_since_last_keyframe >= self.local_map_config.keyframe_min_accepted_frames
            ):
                should_insert = (
                    rotation_angle_deg >= self.local_map_config.keyframe_rotation_thresh_deg
                    or median_parallax_px >= self.local_map_config.keyframe_parallax_thresh_px
                    or shared_track_count
                    <= latest_keyframe.support_count * self.local_map_config.keyframe_track_overlap_ratio
                )

        if not should_insert:
            return 0

        new_keyframe = KeyframeRecord(
            accepted_frame_index=self.accepted_full_pose_count,
            world_from_camera_optical=np.asarray(self.world_from_camera_optical, dtype=np.float64).copy(),
            world_from_base=np.asarray(world_from_base, dtype=np.float64).copy(),
            track_points_by_id=track_points_by_id,
            support_count=len(track_points_by_id),
        )
        return insert_keyframe(
            keyframes=self.keyframes,
            landmarks=self.landmarks,
            new_keyframe=new_keyframe,
            camera_matrix=self.processing_state.camera_matrix,
            config=self.local_map_config,
            triangulation_min_parallax_px=self.scale_config.triangulation_min_parallax_px,
        )

    def _maybe_apply_homography_gate(self, track_result: TrackResult, relative_pose: RelativePoseEstimate) -> None:
        suspicious_frame = (
            relative_pose.health.median_parallax_px < self.min_parallax_px
            or relative_pose.num_inliers < max(self.min_inliers + 15, int(self.min_inliers * 1.25))
        )
        if not self.enable_homography_gate or not suspicious_frame:
            return

        homography_inliers, homography_dominant = evaluate_homography_support(
            previous_points=track_result.previous_points,
            current_points=track_result.current_points,
            ransac_threshold_px=self.homography_ransac_threshold_px,
            essential_inliers=relative_pose.num_inliers,
            dominance_ratio=self.homography_dominance_ratio,
        )
        relative_pose.health.homography_inliers = homography_inliers
        relative_pose.health.homography_dominant = homography_dominant

    def _decide_motion(self, relative_pose: RelativePoseEstimate) -> MotionDecision:
        health = relative_pose.health
        if relative_pose.num_inliers < self.min_inliers:
            return MotionDecision(kind="reject_hold_pose", reason="too_few_inliers")

        if not np.isfinite(health.median_epipolar_error_px):
            return MotionDecision(kind="reject_hold_pose", reason="epipolar_invalid")

        if health.median_epipolar_error_px > self.max_median_epipolar_error_px:
            return MotionDecision(kind="reject_hold_pose", reason="epipolar_too_high")

        if health.median_parallax_px < self.min_parallax_px:
            if health.rotation_angle_deg >= self.yaw_only_min_rotation_deg:
                return MotionDecision(kind="accept_yaw_only", reason="low_parallax")
            return MotionDecision(kind="reject_hold_pose", reason="low_parallax_weak_rotation")

        if health.homography_dominant:
            if health.rotation_angle_deg >= self.yaw_only_min_rotation_deg:
                return MotionDecision(kind="accept_yaw_only", reason="homography_dominant")
            return MotionDecision(kind="reject_hold_pose", reason="homography_dominant_weak_rotation")

        return MotionDecision(kind="accept_full_pose", reason="healthy_geometry")

    def _handle_accept(
        self,
        grayscale: np.ndarray,
        image_msg: Image,
        track_result: TrackResult,
        relative_pose: RelativePoseEstimate,
        survivor_track_ids: np.ndarray,
        decision: MotionDecision,
        yaw_only: bool,
    ) -> None:
        inlier_previous_points = track_result.previous_points[relative_pose.inlier_mask]
        inlier_current_points = track_result.current_points[relative_pose.inlier_mask]
        inlier_track_ids = survivor_track_ids[relative_pose.inlier_mask]
        rejected_previous_points = _merge_points(
            track_result.rejected_previous_points,
            track_result.previous_points[~relative_pose.inlier_mask],
        )
        rejected_current_points = _merge_points(
            track_result.rejected_current_points,
            track_result.current_points[~relative_pose.inlier_mask],
        )

        if yaw_only:
            scale_estimate = hold_scale_estimate(
                last_stable_scale_m=self.last_stable_scale_m,
                bootstrap_scale_m=self.translation_step_scale,
                reason="yaw_only_scale_hold",
            )
        else:
            scale_estimate = self._estimate_scale_for_frame(
                previous_points=inlier_previous_points,
                current_points=inlier_current_points,
                track_ids=inlier_track_ids,
                image_height=grayscale.shape[0],
                relative_pose=relative_pose,
            )

        if yaw_only:
            (
                self.world_from_camera_optical,
                world_from_base,
                vehicle_projection_used,
            ) = self._integrate_yaw_only_step(relative_pose)
            current_step_length_m = 0.0
            refinement_result = PoseRefinementResult(
                attempted=False,
                accepted=False,
                status="yaw_only_skip",
                refined_planar_pose=self.last_planar_pose,
                visible_landmark_count=0,
                grid_cell_count=0,
                rmse_before=float("inf"),
                rmse_after=float("inf"),
                matched_track_ids=np.empty((0,), dtype=np.int64),
                residual_norms_px=np.empty((0,), dtype=np.float64),
            )
        else:
            (
                self.world_from_camera_optical,
                world_from_base,
                vehicle_projection_used,
                current_step_length_m,
            ) = self._integrate_full_pose_step(relative_pose, scale_estimate)
            if scale_estimate.scale_updated:
                self.last_stable_scale_m = scale_estimate.filtered_step_scale_m
            self.accepted_full_pose_count += 1
            self.last_planar_pose = planar_pose_from_world_transform(world_from_base)
            refinement_result = self._maybe_refine_current_pose(
                current_track_ids=survivor_track_ids,
                current_points=track_result.current_points,
                image_shape=grayscale.shape[:2],
                current_step_length_m=current_step_length_m,
            )
            world_from_base = world_from_base_transform(
                world_from_camera_optical=self.world_from_camera_optical,
                base_to_camera_optical=self.base_to_camera_optical,
            )

        self.last_scale_estimate = scale_estimate
        self.last_planar_pose = planar_pose_from_world_transform(world_from_base)
        self.path_msg.poses.append(self._pose_stamped_from_planar_pose(image_msg, self.last_planar_pose))

        self.previous_gray = grayscale
        live_points = inlier_current_points
        live_track_ids = inlier_track_ids
        self._update_live_track_states(live_track_ids, live_points)
        reseeded = False
        if live_points.shape[0] < self.orb_reseed_threshold:
            live_points, live_track_ids, reseeded = self._reseed_points(
                grayscale,
                live_points=live_points,
                live_track_ids=live_track_ids,
                use_orb_first=True,
            )
        elif live_points.shape[0] < self.tracking_config.min_tracked_points:
            live_points, live_track_ids, reseeded = self._reseed_points(
                grayscale,
                live_points=live_points,
                live_track_ids=live_track_ids,
                use_orb_first=False,
            )
        self.previous_points = live_points
        self.previous_track_ids = live_track_ids
        self.consecutive_reject_count = 0
        if not yaw_only:
            self._maybe_insert_keyframe(
                live_track_ids=live_track_ids,
                live_points=live_points,
                world_from_base=world_from_base,
                rotation_angle_deg=relative_pose.health.rotation_angle_deg,
                median_parallax_px=relative_pose.health.median_parallax_px,
            )

        self.last_track_overlay = DebugTrackOverlay(
            accepted_previous_points=inlier_previous_points,
            accepted_current_points=inlier_current_points,
            rejected_previous_points=rejected_previous_points,
            rejected_current_points=rejected_current_points,
            decision=decision.kind,
            reason=decision.reason,
            total_input_tracks=track_result.health.total_input_points,
            forward_survivors=track_result.health.forward_survivor_count,
            fb_survivors=track_result.health.fb_survivor_count,
            inlier_count=relative_pose.num_inliers,
            median_fb_error_px=track_result.health.median_fb_error_px,
            median_epipolar_error_px=relative_pose.health.median_epipolar_error_px,
            median_parallax_px=relative_pose.health.median_parallax_px,
            homography_inliers=relative_pose.health.homography_inliers,
            rotation_angle_deg=relative_pose.health.rotation_angle_deg,
            reseeded=reseeded,
            scale_candidate_count=scale_estimate.candidate_count,
            triangulated_count=scale_estimate.triangulated_count,
            plane_inlier_count=scale_estimate.plane_inlier_count,
            plane_inlier_ratio=scale_estimate.plane_inlier_ratio,
            raw_step_scale_m=scale_estimate.raw_step_scale_m,
            filtered_step_scale_m=scale_estimate.filtered_step_scale_m,
            scale_confidence=scale_estimate.confidence,
            scale_status=scale_estimate.reason,
            vehicle_projection_used=vehicle_projection_used,
            active_keyframes=len(self.keyframes),
            active_landmarks=len(self.landmarks),
            visible_landmark_count=refinement_result.visible_landmark_count,
            refinement_attempted=refinement_result.attempted,
            refinement_applied=refinement_result.accepted,
            refinement_rmse_before=refinement_result.rmse_before,
            refinement_rmse_after=refinement_result.rmse_after,
            refinement_status=refinement_result.status,
        )

    def _handle_reject_hold(
        self,
        grayscale: np.ndarray,
        track_result: TrackResult,
        relative_pose: RelativePoseEstimate | None,
        reason: str,
    ) -> None:
        self.previous_gray = grayscale
        self.consecutive_reject_count += 1
        use_orb_first = self.consecutive_reject_count >= self.consecutive_rejects_for_reseed
        self.previous_points, self.previous_track_ids, reseeded = self._reseed_points(
            grayscale,
            live_points=_empty_points(),
            live_track_ids=np.empty((0,), dtype=np.int64),
            use_orb_first=use_orb_first,
        )
        self.last_scale_estimate = hold_scale_estimate(
            last_stable_scale_m=self.last_stable_scale_m,
            bootstrap_scale_m=self.translation_step_scale,
            reason="reject_hold_scale_hold",
        )

        if relative_pose is None:
            rejected_previous_points = _merge_points(
                track_result.rejected_previous_points,
                track_result.previous_points,
            )
            rejected_current_points = _merge_points(
                track_result.rejected_current_points,
                track_result.current_points,
            )
            inlier_count = 0
            median_epipolar_error_px = float("inf")
            median_parallax_px = 0.0
            homography_inliers = 0
            rotation_angle_deg = 0.0
        else:
            rejected_previous_points = _merge_points(
                track_result.rejected_previous_points,
                track_result.previous_points,
            )
            rejected_current_points = _merge_points(
                track_result.rejected_current_points,
                track_result.current_points,
            )
            inlier_count = relative_pose.num_inliers
            median_epipolar_error_px = relative_pose.health.median_epipolar_error_px
            median_parallax_px = relative_pose.health.median_parallax_px
            homography_inliers = relative_pose.health.homography_inliers
            rotation_angle_deg = relative_pose.health.rotation_angle_deg

        self.last_track_overlay = DebugTrackOverlay(
            accepted_previous_points=_empty_points(),
            accepted_current_points=_empty_points(),
            rejected_previous_points=rejected_previous_points,
            rejected_current_points=rejected_current_points,
            decision="reject_hold_pose",
            reason=reason,
            total_input_tracks=track_result.health.total_input_points,
            forward_survivors=track_result.health.forward_survivor_count,
            fb_survivors=track_result.health.fb_survivor_count,
            inlier_count=inlier_count,
            median_fb_error_px=track_result.health.median_fb_error_px,
            median_epipolar_error_px=median_epipolar_error_px,
            median_parallax_px=median_parallax_px,
            homography_inliers=homography_inliers,
            rotation_angle_deg=rotation_angle_deg,
            reseeded=reseeded,
            scale_candidate_count=0,
            triangulated_count=0,
            plane_inlier_count=0,
            plane_inlier_ratio=0.0,
            raw_step_scale_m=self.last_scale_estimate.raw_step_scale_m,
            filtered_step_scale_m=self.last_scale_estimate.filtered_step_scale_m,
            scale_confidence=self.last_scale_estimate.confidence,
            scale_status=self.last_scale_estimate.reason,
            vehicle_projection_used=False,
            active_keyframes=len(self.keyframes),
            active_landmarks=len(self.landmarks),
            visible_landmark_count=0,
            refinement_attempted=False,
            refinement_applied=False,
            refinement_rmse_before=float("inf"),
            refinement_rmse_after=float("inf"),
            refinement_status="reject_hold",
        )

    def _reseed_points(
        self,
        grayscale: np.ndarray,
        live_points: np.ndarray,
        live_track_ids: np.ndarray,
        use_orb_first: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        points = live_points.reshape(-1, 2).astype(np.float32) if live_points.size > 0 else _empty_points()
        track_ids = live_track_ids.reshape(-1).astype(np.int64) if live_track_ids.size > 0 else np.empty((0,), dtype=np.int64)
        reseeded = False

        if use_orb_first and points.shape[0] < self.tracking_config.max_features:
            orb_budget = min(
                self.tracking_config.orb_max_features,
                self.tracking_config.max_features - points.shape[0],
            )
            orb_points = detect_orb_features(
                grayscale,
                self.tracking_config,
                existing_points=points,
                max_features=orb_budget,
            )
            if orb_points.size > 0:
                points = _merge_points(points, orb_points)
                track_ids = np.concatenate([track_ids, self._assign_new_track_ids(orb_points.shape[0])])
                reseeded = True

        if points.shape[0] < self.tracking_config.max_features:
            shi_points = detect_features(
                grayscale,
                self.tracking_config,
                existing_points=points,
                max_features=self.tracking_config.max_features - points.shape[0],
            )
            if shi_points.size > 0:
                points = _merge_points(points, shi_points)
                track_ids = np.concatenate([track_ids, self._assign_new_track_ids(shi_points.shape[0])])
                reseeded = True

        points = points[: self.tracking_config.max_features]
        track_ids = track_ids[: self.tracking_config.max_features]
        self.track_states = self._build_track_states(track_ids, points, preserve_existing=True)
        return points, track_ids, reseeded

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

    def _publish_debug_image(self, image_msg: Image, display_bgr: np.ndarray) -> None:
        debug_frame = display_bgr.copy()
        max_tracks_to_draw = 300

        if self.last_track_overlay.decision == "accept_yaw_only":
            accepted_color = (0, 255, 255)
        else:
            accepted_color = (0, 255, 0)

        rejected_count = min(int(self.last_track_overlay.rejected_previous_points.shape[0]), max_tracks_to_draw)
        for index in range(rejected_count):
            previous_point = self.last_track_overlay.rejected_previous_points[index]
            current_point = self.last_track_overlay.rejected_current_points[index]
            start = tuple(np.round(previous_point).astype(int))
            end = tuple(np.round(current_point).astype(int))
            cv2.line(debug_frame, start, end, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(debug_frame, end, 2, (0, 0, 255), -1, cv2.LINE_AA)

        accepted_count = min(int(self.last_track_overlay.accepted_previous_points.shape[0]), max_tracks_to_draw)
        for index in range(accepted_count):
            previous_point = self.last_track_overlay.accepted_previous_points[index]
            current_point = self.last_track_overlay.accepted_current_points[index]
            start = tuple(np.round(previous_point).astype(int))
            end = tuple(np.round(current_point).astype(int))
            cv2.line(debug_frame, start, end, accepted_color, 1, cv2.LINE_AA)
            cv2.circle(debug_frame, end, 2, accepted_color, -1, cv2.LINE_AA)

        lines = [
            "Task 8 Monocular VO Phase 3",
            f"Frame {self.frame_counter}",
            f"Decision {self.last_track_overlay.decision}",
            f"Reason {self.last_track_overlay.reason}",
            (
                f"Tracks {self.last_track_overlay.total_input_tracks}  "
                f"Fwd {self.last_track_overlay.forward_survivors}  "
                f"FB {self.last_track_overlay.fb_survivors}  "
                f"Inliers {self.last_track_overlay.inlier_count}"
            ),
            (
                f"FBerr {self._format_metric(self.last_track_overlay.median_fb_error_px)}  "
                f"Epi {self._format_metric(self.last_track_overlay.median_epipolar_error_px)}  "
                f"Parallax {self.last_track_overlay.median_parallax_px:.2f}px"
            ),
            (
                f"Rot {self.last_track_overlay.rotation_angle_deg:.2f} deg  "
                f"Homography {self.last_track_overlay.homography_inliers}  "
                f"Rejects {self.consecutive_reject_count}"
            ),
            (
                f"Scale status {self.last_track_overlay.scale_status}  "
                f"Cand {self.last_track_overlay.scale_candidate_count}  "
                f"Tri {self.last_track_overlay.triangulated_count}  "
                f"Plane {self.last_track_overlay.plane_inlier_count}"
            ),
            (
                f"Raw scale {self._format_optional_metric(self.last_track_overlay.raw_step_scale_m)}  "
                f"Filtered {self.last_track_overlay.filtered_step_scale_m:.3f}m  "
                f"Conf {self.last_track_overlay.scale_confidence:.2f}  "
                f"PlaneRatio {self.last_track_overlay.plane_inlier_ratio:.2f}"
            ),
            (
                f"VehicleProj {'yes' if self.last_track_overlay.vehicle_projection_used else 'no'}  "
                f"Keyframes {self.last_track_overlay.active_keyframes}  "
                f"Landmarks {self.last_track_overlay.active_landmarks}  "
                f"Visible {self.last_track_overlay.visible_landmark_count}"
            ),
            (
                f"Refine {self.last_track_overlay.refinement_status}  "
                f"Attempt {'yes' if self.last_track_overlay.refinement_attempted else 'no'}  "
                f"Applied {'yes' if self.last_track_overlay.refinement_applied else 'no'}  "
                f"RMSE {self._format_metric(self.last_track_overlay.refinement_rmse_before)}->"
                f"{self._format_metric(self.last_track_overlay.refinement_rmse_after)}"
            ),
            (
                f"Pose x={self.last_planar_pose.x:.2f} y={self.last_planar_pose.y:.2f} "
                f"yaw={np.degrees(self.last_planar_pose.yaw):.1f} deg  "
                f"Reseed {'yes' if self.last_track_overlay.reseeded else 'no'}"
            ),
        ]

        for index, text in enumerate(lines):
            cv2.putText(
                debug_frame,
                text,
                (20, 38 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
        debug_msg.header.stamp = image_msg.header.stamp
        debug_msg.header.frame_id = image_msg.header.frame_id
        self.debug_pub.publish(debug_msg)

    @staticmethod
    def _format_metric(value: float) -> str:
        if not np.isfinite(value):
            return "inf"
        return f"{value:.2f}"

    @staticmethod
    def _format_optional_metric(value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return "hold"
        return f"{value:.3f}m"


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
