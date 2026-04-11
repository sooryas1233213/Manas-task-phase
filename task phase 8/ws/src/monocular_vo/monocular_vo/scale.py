from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monocular_vo.geometry import triangulate_correspondences


@dataclass(frozen=True)
class ScaleConfig:
    scale_mode: str
    min_scale_track_age: int
    ground_region_min_y_frac: float
    triangulation_min_parallax_px: float
    ground_flow_angle_tolerance_deg: float
    min_scale_candidate_points: int
    min_plane_inliers: int
    min_plane_inlier_ratio: float
    max_ground_normal_deviation_deg: float
    scale_ema_alpha: float
    max_scale_jump_ratio: float
    min_scale_confidence: float
    min_step_scale_m: float
    max_step_scale_m: float
    bootstrap_scale_m: float


@dataclass(frozen=True)
class PlaneFitResult:
    normal: np.ndarray
    offset: float
    inlier_mask: np.ndarray
    inlier_count: int
    inlier_ratio: float
    normal_deviation_deg: float
    valid: bool


@dataclass(frozen=True)
class ScaleEstimate:
    applied_step_scale_m: float
    raw_step_scale_m: float | None
    filtered_step_scale_m: float
    confidence: float
    candidate_count: int
    triangulated_count: int
    plane_inlier_count: int
    plane_inlier_ratio: float
    scale_updated: bool
    used_fallback: bool
    reason: str


def _default_scale_value(last_stable_scale_m: float | None, bootstrap_scale_m: float) -> float:
    if last_stable_scale_m is not None and np.isfinite(last_stable_scale_m):
        return float(last_stable_scale_m)
    return float(bootstrap_scale_m)


def hold_scale_estimate(
    *,
    last_stable_scale_m: float | None,
    bootstrap_scale_m: float,
    reason: str,
) -> ScaleEstimate:
    applied_scale = _default_scale_value(last_stable_scale_m, bootstrap_scale_m)
    return ScaleEstimate(
        applied_step_scale_m=applied_scale,
        raw_step_scale_m=None,
        filtered_step_scale_m=applied_scale,
        confidence=0.0,
        candidate_count=0,
        triangulated_count=0,
        plane_inlier_count=0,
        plane_inlier_ratio=0.0,
        scale_updated=False,
        used_fallback=True,
        reason=reason,
    )


def _wrap_angle_radians(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean_angle_radians(angles: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("Vector norm must be non-zero.")
    return np.asarray(vector, dtype=np.float64).reshape(3) / norm


def _select_scale_candidates(
    *,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    track_ages: np.ndarray,
    image_height: int,
    config: ScaleConfig,
) -> np.ndarray:
    if previous_points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    displacements = current_points - previous_points
    parallax = np.linalg.norm(displacements, axis=1)
    eligible_mask = (
        (track_ages >= int(config.min_scale_track_age))
        & (current_points[:, 1] >= float(config.ground_region_min_y_frac) * float(image_height))
        & (parallax >= float(config.triangulation_min_parallax_px))
    )
    if int(eligible_mask.sum()) < int(config.min_scale_candidate_points):
        return eligible_mask

    candidate_displacements = displacements[eligible_mask]
    flow_angles = np.arctan2(candidate_displacements[:, 1], candidate_displacements[:, 0])
    median_angle = _circular_mean_angle_radians(flow_angles)
    angle_delta = np.abs(_wrap_angle_radians(flow_angles - median_angle))
    angle_mask = angle_delta <= np.radians(float(config.ground_flow_angle_tolerance_deg))

    final_mask = np.zeros(previous_points.shape[0], dtype=bool)
    candidate_indices = np.nonzero(eligible_mask)[0]
    final_mask[candidate_indices[angle_mask]] = True
    return final_mask


def _plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if points.shape[0] < 3:
        return None

    first = points[1] - points[0]
    second = points[2] - points[0]
    normal = np.cross(first, second)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        return None

    normal = normal / normal_norm
    offset = -float(normal @ points[0])
    return normal.astype(np.float64), offset


def _refine_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if points.shape[0] < 3:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        return None

    normal = normal / normal_norm
    offset = -float(normal @ centroid)
    return normal.astype(np.float64), offset


def _adaptive_plane_distance_threshold(points_3d: np.ndarray) -> float:
    if points_3d.shape[0] == 0:
        return 0.05
    median_depth = float(np.median(points_3d[:, 2]))
    median_range = float(np.median(np.linalg.norm(points_3d, axis=1)))
    scale_hint = max(median_depth, median_range, 1.0)
    return max(0.02, 0.03 * scale_hint)


def _fit_ground_plane(
    *,
    points_3d: np.ndarray,
    expected_up_vector: np.ndarray,
    config: ScaleConfig,
    rng: np.random.Generator,
) -> PlaneFitResult:
    if points_3d.shape[0] < 3:
        return PlaneFitResult(
            normal=np.zeros(3, dtype=np.float64),
            offset=0.0,
            inlier_mask=np.zeros((points_3d.shape[0],), dtype=bool),
            inlier_count=0,
            inlier_ratio=0.0,
            normal_deviation_deg=180.0,
            valid=False,
        )

    threshold = _adaptive_plane_distance_threshold(points_3d)
    expected_up = _normalize_vector(expected_up_vector)
    best_mask = np.zeros((points_3d.shape[0],), dtype=bool)
    best_normal = np.zeros(3, dtype=np.float64)
    best_offset = 0.0
    best_inlier_count = 0
    best_median_distance = float("inf")

    for _ in range(100):
        sample_indices = rng.choice(points_3d.shape[0], size=3, replace=False)
        plane = _plane_from_points(points_3d[sample_indices])
        if plane is None:
            continue

        normal, offset = plane
        if float(normal @ expected_up) < 0.0:
            normal = -normal
            offset = -offset

        distances = np.abs(points_3d @ normal + offset)
        inlier_mask = distances <= threshold
        inlier_count = int(inlier_mask.sum())
        if inlier_count == 0:
            continue

        median_distance = float(np.median(distances[inlier_mask]))
        if inlier_count > best_inlier_count or (
            inlier_count == best_inlier_count and median_distance < best_median_distance
        ):
            best_mask = inlier_mask
            best_normal = normal
            best_offset = float(offset)
            best_inlier_count = inlier_count
            best_median_distance = median_distance

    if best_inlier_count < 3:
        return PlaneFitResult(
            normal=np.zeros(3, dtype=np.float64),
            offset=0.0,
            inlier_mask=best_mask,
            inlier_count=best_inlier_count,
            inlier_ratio=0.0 if points_3d.shape[0] == 0 else best_inlier_count / float(points_3d.shape[0]),
            normal_deviation_deg=180.0,
            valid=False,
        )

    refined = _refine_plane(points_3d[best_mask])
    if refined is not None:
        best_normal, best_offset = refined
        if float(best_normal @ expected_up) < 0.0:
            best_normal = -best_normal
            best_offset = -best_offset

        refined_distances = np.abs(points_3d @ best_normal + best_offset)
        best_mask = refined_distances <= threshold
        best_inlier_count = int(best_mask.sum())

    dot_value = float(np.clip(best_normal @ expected_up, -1.0, 1.0))
    normal_deviation_deg = float(np.degrees(np.arccos(dot_value)))
    inlier_ratio = best_inlier_count / float(points_3d.shape[0]) if points_3d.shape[0] > 0 else 0.0
    valid = (
        best_inlier_count >= int(config.min_plane_inliers)
        and inlier_ratio >= float(config.min_plane_inlier_ratio)
        and normal_deviation_deg <= float(config.max_ground_normal_deviation_deg)
    )
    return PlaneFitResult(
        normal=best_normal,
        offset=float(best_offset),
        inlier_mask=best_mask,
        inlier_count=best_inlier_count,
        inlier_ratio=float(inlier_ratio),
        normal_deviation_deg=normal_deviation_deg,
        valid=bool(valid),
    )


def _scale_confidence(
    *,
    candidate_count: int,
    plane_result: PlaneFitResult,
    config: ScaleConfig,
) -> float:
    candidate_factor = min(1.0, float(candidate_count) / max(float(config.min_scale_candidate_points) * 2.0, 1.0))
    normal_factor = max(
        0.0,
        1.0 - (float(plane_result.normal_deviation_deg) / max(float(config.max_ground_normal_deviation_deg), 1e-6)),
    )
    confidence = 0.4 * float(plane_result.inlier_ratio) + 0.3 * candidate_factor + 0.3 * normal_factor
    return float(np.clip(confidence, 0.0, 1.0))


def estimate_ground_plane_scale(
    *,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    track_ages: np.ndarray,
    image_height: int,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation_unit: np.ndarray,
    camera_height_m: float,
    expected_up_vector: np.ndarray,
    config: ScaleConfig,
    last_stable_scale_m: float | None,
    rng: np.random.Generator,
) -> ScaleEstimate:
    fallback_scale = _default_scale_value(last_stable_scale_m, config.bootstrap_scale_m)
    if config.scale_mode != "ground_plane":
        return hold_scale_estimate(
            last_stable_scale_m=last_stable_scale_m,
            bootstrap_scale_m=config.bootstrap_scale_m,
            reason="unsupported_scale_mode",
        )

    candidate_mask = _select_scale_candidates(
        previous_points=previous_points,
        current_points=current_points,
        track_ages=track_ages,
        image_height=image_height,
        config=config,
    )
    candidate_count = int(candidate_mask.sum())
    if candidate_count < int(config.min_scale_candidate_points):
        return ScaleEstimate(
            applied_step_scale_m=fallback_scale,
            raw_step_scale_m=None,
            filtered_step_scale_m=fallback_scale,
            confidence=0.0,
            candidate_count=candidate_count,
            triangulated_count=0,
            plane_inlier_count=0,
            plane_inlier_ratio=0.0,
            scale_updated=False,
            used_fallback=True,
            reason="too_few_scale_candidates",
        )

    candidate_previous = previous_points[candidate_mask]
    candidate_current = current_points[candidate_mask]
    triangulated_points, triangulated_mask = triangulate_correspondences(
        previous_points=candidate_previous,
        current_points=candidate_current,
        camera_matrix=camera_matrix,
        rotation=rotation,
        translation=translation_unit,
    )
    valid_points = triangulated_points[triangulated_mask]
    triangulated_count = int(valid_points.shape[0])
    if triangulated_count < int(config.min_plane_inliers):
        return ScaleEstimate(
            applied_step_scale_m=fallback_scale,
            raw_step_scale_m=None,
            filtered_step_scale_m=fallback_scale,
            confidence=0.0,
            candidate_count=candidate_count,
            triangulated_count=triangulated_count,
            plane_inlier_count=0,
            plane_inlier_ratio=0.0,
            scale_updated=False,
            used_fallback=True,
            reason="too_few_triangulated_points",
        )

    plane_result = _fit_ground_plane(
        points_3d=valid_points,
        expected_up_vector=expected_up_vector,
        config=config,
        rng=rng,
    )
    if not plane_result.valid:
        return ScaleEstimate(
            applied_step_scale_m=fallback_scale,
            raw_step_scale_m=None,
            filtered_step_scale_m=fallback_scale,
            confidence=0.0,
            candidate_count=candidate_count,
            triangulated_count=triangulated_count,
            plane_inlier_count=plane_result.inlier_count,
            plane_inlier_ratio=plane_result.inlier_ratio,
            scale_updated=False,
            used_fallback=True,
            reason="ground_plane_invalid",
        )

    vo_camera_height = abs(float(plane_result.offset))
    raw_scale = float(camera_height_m) / vo_camera_height if vo_camera_height > 1e-9 else float("nan")
    if (not np.isfinite(raw_scale)) or (
        raw_scale < float(config.min_step_scale_m) or raw_scale > float(config.max_step_scale_m)
    ):
        return ScaleEstimate(
            applied_step_scale_m=fallback_scale,
            raw_step_scale_m=None,
            filtered_step_scale_m=fallback_scale,
            confidence=0.0,
            candidate_count=candidate_count,
            triangulated_count=triangulated_count,
            plane_inlier_count=plane_result.inlier_count,
            plane_inlier_ratio=plane_result.inlier_ratio,
            scale_updated=False,
            used_fallback=True,
            reason="raw_scale_out_of_bounds",
        )

    confidence = _scale_confidence(
        candidate_count=candidate_count,
        plane_result=plane_result,
        config=config,
    )
    if confidence < float(config.min_scale_confidence):
        return ScaleEstimate(
            applied_step_scale_m=fallback_scale,
            raw_step_scale_m=raw_scale,
            filtered_step_scale_m=fallback_scale,
            confidence=confidence,
            candidate_count=candidate_count,
            triangulated_count=triangulated_count,
            plane_inlier_count=plane_result.inlier_count,
            plane_inlier_ratio=plane_result.inlier_ratio,
            scale_updated=False,
            used_fallback=True,
            reason="scale_confidence_low",
        )

    if last_stable_scale_m is not None and np.isfinite(last_stable_scale_m):
        stable_scale = float(last_stable_scale_m)
        jump_ratio = max(raw_scale / stable_scale, stable_scale / raw_scale)
        if jump_ratio > float(config.max_scale_jump_ratio):
            return ScaleEstimate(
                applied_step_scale_m=stable_scale,
                raw_step_scale_m=raw_scale,
                filtered_step_scale_m=stable_scale,
                confidence=confidence,
                candidate_count=candidate_count,
                triangulated_count=triangulated_count,
                plane_inlier_count=plane_result.inlier_count,
                plane_inlier_ratio=plane_result.inlier_ratio,
                scale_updated=False,
                used_fallback=True,
                reason="scale_jump_too_large",
            )
        filtered_scale = float(config.scale_ema_alpha) * raw_scale + (1.0 - float(config.scale_ema_alpha)) * stable_scale
    else:
        filtered_scale = raw_scale

    return ScaleEstimate(
        applied_step_scale_m=filtered_scale,
        raw_step_scale_m=raw_scale,
        filtered_step_scale_m=filtered_scale,
        confidence=confidence,
        candidate_count=candidate_count,
        triangulated_count=triangulated_count,
        plane_inlier_count=plane_result.inlier_count,
        plane_inlier_ratio=plane_result.inlier_ratio,
        scale_updated=True,
        used_fallback=False,
        reason="ground_plane_scale_updated",
    )
