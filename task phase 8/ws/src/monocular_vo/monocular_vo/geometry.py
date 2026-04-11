from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class RelativePoseEstimate:
    current_from_previous: np.ndarray
    inlier_mask: np.ndarray
    num_inliers: int
    rotation: np.ndarray
    translation: np.ndarray


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
    translation_scale: float,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3) * float(translation_scale)
    return transform


def estimate_relative_pose(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    camera_matrix: np.ndarray,
    ransac_threshold_px: float,
    translation_step_scale: float,
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
    return RelativePoseEstimate(
        current_from_previous=_transform_from_rotation_translation(
            rotation=rotation,
            translation=translation.reshape(3),
            translation_scale=translation_step_scale,
        ),
        inlier_mask=inlier_mask_bool,
        num_inliers=int(num_inliers),
        rotation=np.asarray(rotation, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64).reshape(3),
    )
