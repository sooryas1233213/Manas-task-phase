from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class GeometryHealth:
    essential_inliers: int
    median_epipolar_error_px: float
    median_parallax_px: float
    rotation_angle_deg: float
    homography_inliers: int = 0
    homography_dominant: bool = False


@dataclass
class RelativePoseEstimate:
    current_from_previous: np.ndarray
    inlier_mask: np.ndarray
    num_inliers: int
    rotation: np.ndarray
    translation: np.ndarray
    health: GeometryHealth


def _coerce_essential_matrix(essential_matrix: np.ndarray | None) -> Optional[np.ndarray]:
    if essential_matrix is None:
        return None

    essential_matrix = np.asarray(essential_matrix, dtype=np.float64)
    if essential_matrix.shape == (3, 3):
        return essential_matrix
    if essential_matrix.ndim == 2 and essential_matrix.shape[1] == 3 and essential_matrix.shape[0] >= 3:
        return essential_matrix[:3, :]
    return None


def _transform_from_rotation_translation(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    trace_value = float(np.trace(rotation))
    cosine_value = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_value)))


def _median_parallax_px(previous_points: np.ndarray, current_points: np.ndarray) -> float:
    if previous_points.size == 0 or current_points.size == 0:
        return 0.0
    displacements = np.linalg.norm(current_points - previous_points, axis=1)
    return float(np.median(displacements)) if displacements.size > 0 else 0.0


def _median_symmetric_epipolar_error_px(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    essential_matrix: np.ndarray,
    camera_matrix: np.ndarray,
) -> float:
    if previous_points.shape[0] == 0 or current_points.shape[0] == 0:
        return float("inf")

    homogeneous_previous = np.hstack([previous_points, np.ones((previous_points.shape[0], 1), dtype=np.float64)])
    homogeneous_current = np.hstack([current_points, np.ones((current_points.shape[0], 1), dtype=np.float64)])
    camera_matrix_inverse = np.linalg.inv(camera_matrix)
    fundamental_matrix = camera_matrix_inverse.T @ essential_matrix @ camera_matrix_inverse

    lines_in_current = (fundamental_matrix @ homogeneous_previous.T).T
    lines_in_previous = (fundamental_matrix.T @ homogeneous_current.T).T
    numerators = np.abs(np.sum(homogeneous_current * lines_in_current, axis=1))
    denom_current = np.linalg.norm(lines_in_current[:, :2], axis=1)
    denom_previous = np.linalg.norm(lines_in_previous[:, :2], axis=1)
    valid_mask = (denom_current > 1e-9) & (denom_previous > 1e-9)
    if not np.any(valid_mask):
        return float("inf")

    symmetric_errors = 0.5 * numerators[valid_mask] * (
        (1.0 / denom_current[valid_mask]) + (1.0 / denom_previous[valid_mask])
    )
    return float(np.median(symmetric_errors)) if symmetric_errors.size > 0 else float("inf")


def estimate_relative_pose(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    camera_matrix: np.ndarray,
    ransac_threshold_px: float,
) -> Optional[RelativePoseEstimate]:
    if previous_points.shape[0] < 8 or current_points.shape[0] < 8:
        return None

    essential_matrix, essential_mask = cv2.findEssentialMat(
        previous_points,
        current_points,
        cameraMatrix=camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=float(ransac_threshold_px),
    )
    essential_matrix = _coerce_essential_matrix(essential_matrix)
    if essential_matrix is None or essential_mask is None:
        return None

    num_inliers, rotation, translation, inlier_mask = cv2.recoverPose(
        essential_matrix,
        previous_points,
        current_points,
        cameraMatrix=camera_matrix,
        mask=essential_mask,
    )
    if inlier_mask is None:
        return None

    inlier_mask_bool = inlier_mask.reshape(-1).astype(bool)
    inlier_previous_points = previous_points[inlier_mask_bool]
    inlier_current_points = current_points[inlier_mask_bool]
    health = GeometryHealth(
        essential_inliers=int(num_inliers),
        median_epipolar_error_px=_median_symmetric_epipolar_error_px(
            previous_points=inlier_previous_points,
            current_points=inlier_current_points,
            essential_matrix=essential_matrix,
            camera_matrix=camera_matrix,
        ),
        median_parallax_px=_median_parallax_px(
            previous_points=inlier_previous_points,
            current_points=inlier_current_points,
        ),
        rotation_angle_deg=_rotation_angle_degrees(rotation),
    )

    return RelativePoseEstimate(
        current_from_previous=_transform_from_rotation_translation(
            rotation=rotation,
            translation=translation.reshape(3),
        ),
        inlier_mask=inlier_mask_bool,
        num_inliers=int(num_inliers),
        rotation=np.asarray(rotation, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64).reshape(3),
        health=health,
    )


def projection_matrices_from_relative_pose(
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    projection_previous = camera_matrix @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    projection_current = camera_matrix @ np.hstack(
        [
            np.asarray(rotation, dtype=np.float64).reshape(3, 3),
            np.asarray(translation, dtype=np.float64).reshape(3, 1),
        ]
    )
    return projection_previous.astype(np.float64), projection_current.astype(np.float64)


def _reprojection_errors_px(
    points_3d_previous: np.ndarray,
    image_points: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    homogeneous_points = np.hstack(
        [np.asarray(points_3d_previous, dtype=np.float64), np.ones((points_3d_previous.shape[0], 1), dtype=np.float64)]
    )
    projected = (projection_matrix @ homogeneous_points.T).T
    valid_depth = np.abs(projected[:, 2]) > 1e-9
    image_estimates = np.full((points_3d_previous.shape[0], 2), np.nan, dtype=np.float64)
    image_estimates[valid_depth] = projected[valid_depth, :2] / projected[valid_depth, 2:3]
    return np.linalg.norm(image_estimates - np.asarray(image_points, dtype=np.float64), axis=1)


def triangulate_correspondences(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    max_reprojection_error_px: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    if previous_points.shape[0] == 0 or current_points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)

    projection_previous, projection_current = projection_matrices_from_relative_pose(
        camera_matrix=camera_matrix,
        rotation=rotation,
        translation=translation,
    )
    homogeneous_points = cv2.triangulatePoints(
        projection_previous,
        projection_current,
        previous_points.T,
        current_points.T,
    )
    points_3d_previous = cv2.convertPointsFromHomogeneous(homogeneous_points.T).reshape(-1, 3).astype(np.float64)
    depth_previous = points_3d_previous[:, 2]
    points_3d_current = (
        np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ points_3d_previous.T
        + np.asarray(translation, dtype=np.float64).reshape(3, 1)
    ).T
    depth_current = points_3d_current[:, 2]
    positive_depth_mask = (depth_previous > 1e-6) & (depth_current > 1e-6)

    if max_reprojection_error_px <= 0.0:
        return points_3d_previous, positive_depth_mask

    previous_reprojection_errors = _reprojection_errors_px(
        points_3d_previous=points_3d_previous,
        image_points=previous_points,
        projection_matrix=projection_previous,
    )
    current_reprojection_errors = _reprojection_errors_px(
        points_3d_previous=points_3d_previous,
        image_points=current_points,
        projection_matrix=projection_current,
    )
    reprojection_mask = (
        np.isfinite(previous_reprojection_errors)
        & np.isfinite(current_reprojection_errors)
        & (previous_reprojection_errors <= float(max_reprojection_error_px))
        & (current_reprojection_errors <= float(max_reprojection_error_px))
    )
    return points_3d_previous, positive_depth_mask & reprojection_mask


def evaluate_homography_support(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    ransac_threshold_px: float,
    essential_inliers: int,
    dominance_ratio: float,
) -> tuple[int, bool]:
    if previous_points.shape[0] < 4 or current_points.shape[0] < 4:
        return 0, False

    _, homography_mask = cv2.findHomography(
        previous_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_px),
    )
    if homography_mask is None:
        return 0, False

    homography_inliers = int(homography_mask.reshape(-1).sum())
    homography_dominant = homography_inliers > float(essential_inliers) * float(dominance_ratio)
    return homography_inliers, homography_dominant
