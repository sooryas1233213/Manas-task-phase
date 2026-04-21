from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def base_to_camera_optical_transform(
    camera_translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(-0.5, 0.5, -0.5, 0.5)
    return make_transform(rotation, np.array(camera_translation_xyz, dtype=np.float64))


def rotation_matrix_from_yaw(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def initial_world_from_camera_optical(
    camera_translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    return base_to_camera_optical_transform(camera_translation_xyz)


def world_from_planar_base_pose(planar_pose: PlanarPose) -> np.ndarray:
    return make_transform(
        rotation_matrix_from_yaw(planar_pose.yaw),
        np.array([planar_pose.x, planar_pose.y, 0.0], dtype=np.float64),
    )


def integrate_camera_motion(
    world_from_camera_optical: np.ndarray,
    current_from_previous: np.ndarray,
) -> np.ndarray:
    return world_from_camera_optical @ invert_transform(current_from_previous)


def integrate_base_motion(
    world_from_base: np.ndarray,
    current_from_previous_base: np.ndarray,
) -> np.ndarray:
    return world_from_base @ invert_transform(current_from_previous_base)


def scaled_camera_motion_transform(
    rotation: np.ndarray,
    translation_unit: np.ndarray,
    step_scale_m: float,
) -> np.ndarray:
    return make_transform(
        np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        np.asarray(translation_unit, dtype=np.float64).reshape(3) * float(step_scale_m),
    )


def zero_translation_transform(transform: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform[:3, :3], dtype=np.float64).reshape(3, 3)
    return make_transform(rotation, np.zeros(3, dtype=np.float64))


def current_base_from_previous_base_transform(
    current_from_previous_camera: np.ndarray,
    base_to_camera_optical: np.ndarray,
) -> np.ndarray:
    camera_from_base = invert_transform(base_to_camera_optical)
    return base_to_camera_optical @ current_from_previous_camera @ camera_from_base


def project_base_motion_to_planar(current_from_previous_base: np.ndarray) -> np.ndarray:
    previous_from_current_base = invert_transform(current_from_previous_base)
    yaw_delta = math.atan2(
        float(previous_from_current_base[1, 0]),
        float(previous_from_current_base[0, 0]),
    )
    planar_displacement = previous_from_current_base[:2, 3]
    forward_step_m = float(np.linalg.norm(planar_displacement))
    previous_from_current_planar = make_transform(
        rotation_matrix_from_yaw(yaw_delta),
        np.array([forward_step_m, 0.0, 0.0], dtype=np.float64),
    )
    return invert_transform(previous_from_current_planar)


def project_base_rotation_to_yaw(current_from_previous_base: np.ndarray) -> np.ndarray:
    previous_from_current_base = invert_transform(current_from_previous_base)
    yaw_delta = math.atan2(
        float(previous_from_current_base[1, 0]),
        float(previous_from_current_base[0, 0]),
    )
    previous_from_current_planar = make_transform(
        rotation_matrix_from_yaw(yaw_delta),
        np.zeros(3, dtype=np.float64),
    )
    return invert_transform(previous_from_current_planar)


def planar_step_length_from_relative_base_transform(current_from_previous_base: np.ndarray) -> float:
    previous_from_current_base = invert_transform(current_from_previous_base)
    return float(np.linalg.norm(previous_from_current_base[:2, 3]))


def integrate_camera_rotation_only(
    world_from_camera_optical: np.ndarray,
    current_from_previous: np.ndarray,
) -> np.ndarray:
    return world_from_camera_optical @ invert_transform(zero_translation_transform(current_from_previous))


def world_from_base_transform(
    world_from_camera_optical: np.ndarray,
    base_to_camera_optical: np.ndarray,
) -> np.ndarray:
    return world_from_camera_optical @ invert_transform(base_to_camera_optical)


def planar_pose_from_world_transform(world_from_base: np.ndarray) -> PlanarPose:
    forward_axis_world = world_from_base[:3, 0]
    yaw = math.atan2(float(forward_axis_world[1]), float(forward_axis_world[0]))
    return PlanarPose(
        x=float(world_from_base[0, 3]),
        y=float(world_from_base[1, 3]),
        yaw=float(yaw),
    )


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
