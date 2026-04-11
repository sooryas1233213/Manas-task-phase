from __future__ import annotations

import sys

import numpy as np

from monocular_vo.pose_integration import base_to_camera_optical_transform
from monocular_vo.scale import ScaleConfig, estimate_ground_plane_scale


def _project_points(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    projected = (camera_matrix @ points_3d.T).T
    return (projected[:, :2] / projected[:, 2:3]).astype(np.float32)


def main() -> None:
    image_width = 960
    image_height = 540
    camera_height_m = 1.3
    true_step_scale_m = 0.35
    camera_matrix = np.array(
        [
            [700.0, 0.0, image_width * 0.5],
            [0.0, 700.0, image_height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    base_to_camera = base_to_camera_optical_transform()
    rotation_base_to_camera = base_to_camera[:3, :3]
    expected_up = rotation_base_to_camera @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    expected_up = expected_up / np.linalg.norm(expected_up)
    translation_unit = rotation_base_to_camera @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    translation_unit = translation_unit / np.linalg.norm(translation_unit)
    translation_actual = translation_unit * true_step_scale_m

    forward_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    plane_axis_u = forward_hint - expected_up * float(expected_up @ forward_hint)
    if np.linalg.norm(plane_axis_u) < 1e-6:
        forward_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        plane_axis_u = forward_hint - expected_up * float(expected_up @ forward_hint)
    plane_axis_u = plane_axis_u / np.linalg.norm(plane_axis_u)
    plane_axis_v = np.cross(expected_up, plane_axis_u)
    plane_axis_v = plane_axis_v / np.linalg.norm(plane_axis_v)

    forward_values, lateral_values = np.meshgrid(
        np.linspace(8.0, 28.0, 16),
        np.linspace(-3.0, 6.0, 12),
    )
    plane_origin = expected_up * camera_height_m
    previous_camera_points = (
        plane_origin
        + forward_values.reshape(-1, 1) * plane_axis_u
        + lateral_values.reshape(-1, 1) * plane_axis_v
    )
    current_camera_points = previous_camera_points + translation_actual
    valid_depth = (previous_camera_points[:, 2] > 0.5) & (current_camera_points[:, 2] > 0.5)
    previous_camera_points = previous_camera_points[valid_depth]
    current_camera_points = current_camera_points[valid_depth]

    previous_pixels = _project_points(previous_camera_points, camera_matrix)
    current_pixels = _project_points(current_camera_points, camera_matrix)
    in_bounds = (
        (previous_pixels[:, 0] >= 0.0)
        & (previous_pixels[:, 0] < image_width)
        & (previous_pixels[:, 1] >= 0.0)
        & (previous_pixels[:, 1] < image_height)
        & (current_pixels[:, 0] >= 0.0)
        & (current_pixels[:, 0] < image_width)
        & (current_pixels[:, 1] >= 0.0)
        & (current_pixels[:, 1] < image_height)
    )
    previous_pixels = previous_pixels[in_bounds]
    current_pixels = current_pixels[in_bounds]

    config = ScaleConfig(
        scale_mode="ground_plane",
        min_scale_track_age=3,
        ground_region_min_y_frac=0.55,
        triangulation_min_parallax_px=3.0,
        ground_flow_angle_tolerance_deg=20.0,
        min_scale_candidate_points=30,
        min_plane_inliers=20,
        min_plane_inlier_ratio=0.5,
        max_ground_normal_deviation_deg=35.0,
        scale_ema_alpha=0.25,
        max_scale_jump_ratio=1.5,
        min_scale_confidence=0.55,
        min_step_scale_m=0.01,
        max_step_scale_m=5.0,
        bootstrap_scale_m=1.0,
    )

    estimate = estimate_ground_plane_scale(
        previous_points=previous_pixels,
        current_points=current_pixels,
        track_ages=np.full(previous_pixels.shape[0], 3, dtype=np.int32),
        image_height=image_height,
        camera_matrix=camera_matrix,
        rotation=np.eye(3, dtype=np.float64),
        translation_unit=translation_unit,
        camera_height_m=camera_height_m,
        expected_up_vector=expected_up,
        config=config,
        last_stable_scale_m=None,
        rng=np.random.default_rng(7),
    )

    error = abs(estimate.applied_step_scale_m - true_step_scale_m)
    print(
        "Phase 3 scale sanity:",
        f"applied={estimate.applied_step_scale_m:.4f}m",
        f"true={true_step_scale_m:.4f}m",
        f"error={error:.4f}m",
        f"confidence={estimate.confidence:.2f}",
        f"reason={estimate.reason}",
    )

    if estimate.scale_updated and error <= 0.08:
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()
