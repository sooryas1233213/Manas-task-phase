from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from monocular_vo.geometry import triangulate_correspondences
from monocular_vo.pose_integration import PlanarPose, invert_transform, world_from_planar_base_pose

_MIN_GRID_CELLS = 6
_MIN_DEPTH_M = 1e-6


@dataclass(frozen=True)
class LocalMapCfg:
    max_keyframes: int
    keyframe_min_accepted_frames: int
    kf_force_frames: int
    keyframe_rotation_thresh_deg: float
    keyframe_parallax_thresh_px: float
    keyframe_track_overlap_ratio: float
    min_landmarks: int
    max_reproj_px: float
    max_iters: int
    max_yaw_deg: float
    max_step_ratio: float


@dataclass(frozen=True)
class KeyframeRec:
    frame_idx: int
    world_from_camera_optical: np.ndarray
    world_from_base: np.ndarray
    track_points_by_id: dict[int, np.ndarray]
    support_count: int


@dataclass(frozen=True)
class LandmarkRec:
    track_id: int
    world_point: np.ndarray
    src_kf_indices: tuple[int, int]
    last_reproj_px: float
    support_count: int
    last_seen_frame_idx: int


@dataclass(frozen=True)
class RefineResult:
    attempted: bool
    accepted: bool
    status: str
    refined_planar_pose: PlanarPose
    vis_lm_count: int
    grid_cells: int
    rmse_before: float
    rmse_after: float
    matched_track_ids: np.ndarray
    resid_px: np.ndarray


def insert_keyframe(
    keyframes: deque[KeyframeRec],
    landmarks: dict[int, LandmarkRec],
    new_keyframe: KeyframeRec,
    camera_matrix: np.ndarray,
    config: LocalMapCfg,
    triangulation_min_parallax_px: float,
) -> int:
    prev_kf = keyframes[-1] if keyframes else None
    keyframes.append(new_keyframe)

    added_lms = {}
    if prev_kf is not None:
        added_lms = triangulate_new_landmarks(
            previous_keyframe=prev_kf,
            current_keyframe=new_keyframe,
            camera_matrix=camera_matrix,
            max_reprojection_error_px=config.max_reproj_px,
            min_parallax_px=triangulation_min_parallax_px,
        )
        landmarks.update(added_lms)

    while len(keyframes) > config.max_keyframes:
        keyframes.popleft()

    active_kf_indices = {keyframe.frame_idx for keyframe in keyframes}
    stale_track_ids = [
        track_id
        for track_id, landmark in landmarks.items()
        if any(index not in active_kf_indices for index in landmark.src_kf_indices)
        or (
            np.isfinite(landmark.last_reproj_px)
            and landmark.last_reproj_px > config.max_reproj_px * 2.0
        )
    ]
    for track_id in stale_track_ids:
        landmarks.pop(track_id, None)

    return len(added_lms)


def triangulate_new_landmarks(
    previous_keyframe: KeyframeRec,
    current_keyframe: KeyframeRec,
    camera_matrix: np.ndarray,
    max_reprojection_error_px: float,
    min_parallax_px: float,
) -> dict[int, LandmarkRec]:
    shared_track_ids = sorted(
        set(previous_keyframe.track_points_by_id.keys()) & set(current_keyframe.track_points_by_id.keys())
    )
    if not shared_track_ids:
        return {}

    previous_points = np.array(
        [previous_keyframe.track_points_by_id[track_id] for track_id in shared_track_ids],
        dtype=np.float64,
    )
    current_points = np.array(
        [current_keyframe.track_points_by_id[track_id] for track_id in shared_track_ids],
        dtype=np.float64,
    )
    parallax = np.linalg.norm(current_points - previous_points, axis=1)
    parallax_mask = parallax >= float(min_parallax_px)
    if not np.any(parallax_mask):
        return {}

    filtered_track_ids = np.asarray(shared_track_ids, dtype=np.int64)[parallax_mask]
    filtered_previous_points = previous_points[parallax_mask]
    filtered_current_points = current_points[parallax_mask]

    current_from_previous = (
        invert_transform(current_keyframe.world_from_camera_optical) @ previous_keyframe.world_from_camera_optical
    )
    points_3d_previous, valid_mask = triangulate_correspondences(
        previous_points=filtered_previous_points,
        current_points=filtered_current_points,
        camera_matrix=camera_matrix,
        rotation=current_from_previous[:3, :3],
        translation=current_from_previous[:3, 3],
        max_reprojection_error_px=max_reprojection_error_px,
    )
    if not np.any(valid_mask):
        return {}

    world_points = _transform_points(previous_keyframe.world_from_camera_optical, points_3d_previous[valid_mask])
    reprojection_errors = _mean_two_view_reprojection_errors(
        world_points=world_points,
        previous_keyframe=previous_keyframe,
        current_keyframe=current_keyframe,
        previous_points=filtered_previous_points[valid_mask],
        current_points=filtered_current_points[valid_mask],
        camera_matrix=camera_matrix,
    )

    landmarks: dict[int, LandmarkRec] = {}
    for track_id, world_point, reproj_px in zip(
        filtered_track_ids[valid_mask].tolist(),
        world_points,
        reprojection_errors.tolist(),
    ):
        landmarks[int(track_id)] = LandmarkRec(
            track_id=int(track_id),
            world_point=np.asarray(world_point, dtype=np.float64),
            src_kf_indices=(
                previous_keyframe.frame_idx,
                current_keyframe.frame_idx,
            ),
            last_reproj_px=float(reproj_px),
            support_count=2,
            last_seen_frame_idx=current_keyframe.frame_idx,
        )
    return landmarks


def collect_visible_landmarks(
    landmarks: dict[int, LandmarkRec],
    current_track_ids: np.ndarray,
    current_points: np.ndarray,
    image_shape: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if current_track_ids.size == 0 or current_points.size == 0 or not landmarks:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            0,
        )

    point_by_track_id = {
        int(track_id): np.asarray(point, dtype=np.float64)
        for track_id, point in zip(current_track_ids.tolist(), current_points.reshape(-1, 2))
    }
    matched_track_ids: list[int] = []
    world_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    occupied_cells: set[tuple[int, int]] = set()
    height, width = image_shape
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))

    for track_id, landmark in landmarks.items():
        image_point = point_by_track_id.get(int(track_id))
        if image_point is None:
            continue
        matched_track_ids.append(int(track_id))
        world_points.append(np.asarray(landmark.world_point, dtype=np.float64))
        image_points.append(np.asarray(image_point, dtype=np.float64))
        cell_row = min(grid_rows - 1, max(0, int(image_point[1] * grid_rows / safe_height)))
        cell_col = min(grid_cols - 1, max(0, int(image_point[0] * grid_cols / safe_width)))
        occupied_cells.add((cell_row, cell_col))

    if not matched_track_ids:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            0,
        )

    return (
        np.asarray(matched_track_ids, dtype=np.int64),
        np.asarray(world_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        len(occupied_cells),
    )


def refine_current_pose(
    current_planar_pose: PlanarPose,
    matched_track_ids: np.ndarray,
    world_points: np.ndarray,
    image_points: np.ndarray,
    grid_cell_count: int,
    camera_matrix: np.ndarray,
    base_to_camera_optical: np.ndarray,
    current_step_length_m: float,
    config: LocalMapCfg,
) -> RefineResult:
    if matched_track_ids.size < config.min_landmarks:
        return RefineResult(
            attempted=False,
            accepted=False,
            status="too_few_visible_landmarks",
            refined_planar_pose=current_planar_pose,
            vis_lm_count=int(matched_track_ids.size),
            grid_cells=int(grid_cell_count),
            rmse_before=float("inf"),
            rmse_after=float("inf"),
            matched_track_ids=matched_track_ids,
            resid_px=np.empty((0,), dtype=np.float64),
        )
    if grid_cell_count < _MIN_GRID_CELLS:
        return RefineResult(
            attempted=False,
            accepted=False,
            status="insufficient_grid_coverage",
            refined_planar_pose=current_planar_pose,
            vis_lm_count=int(matched_track_ids.size),
            grid_cells=int(grid_cell_count),
            rmse_before=float("inf"),
            rmse_after=float("inf"),
            matched_track_ids=matched_track_ids,
            resid_px=np.empty((0,), dtype=np.float64),
        )

    params = np.array(
        [current_planar_pose.x, current_planar_pose.y, current_planar_pose.yaw],
        dtype=np.float64,
    )
    residual_vector, residual_norms, valid_depth = _residuals_from_planar_pose(
        params=params,
        world_points=world_points,
        image_points=image_points,
        camera_matrix=camera_matrix,
        base_to_camera_optical=base_to_camera_optical,
    )
    if residual_vector is None or not valid_depth:
        return RefineResult(
            attempted=False,
            accepted=False,
            status="invalid_initial_projection",
            refined_planar_pose=current_planar_pose,
            vis_lm_count=int(matched_track_ids.size),
            grid_cells=int(grid_cell_count),
            rmse_before=float("inf"),
            rmse_after=float("inf"),
            matched_track_ids=matched_track_ids,
            resid_px=np.empty((0,), dtype=np.float64),
        )

    rmse_before = float(np.sqrt(np.mean(residual_norms * residual_norms)))
    best_params = params.copy()
    best_rmse = rmse_before
    lambda_diag = 1e-3
    step_eps = np.array([1e-2, 1e-2, np.deg2rad(0.1)], dtype=np.float64)

    for _ in range(config.max_iters):
        residual_vector, residual_norms, valid_depth = _residuals_from_planar_pose(
            params=best_params,
            world_points=world_points,
            image_points=image_points,
            camera_matrix=camera_matrix,
            base_to_camera_optical=base_to_camera_optical,
        )
        if residual_vector is None or not valid_depth:
            break

        huber_weights = _huber_weights(residual_norms, config.max_reproj_px)
        sqrt_weights = np.repeat(np.sqrt(huber_weights), 2)
        weighted_residual_vector = residual_vector * sqrt_weights

        jacobian_columns = []
        for parameter_index in range(3):
            perturbed_params = best_params.copy()
            perturbed_params[parameter_index] += step_eps[parameter_index]
            perturbed_residual_vector, _, perturbed_valid_depth = _residuals_from_planar_pose(
                params=perturbed_params,
                world_points=world_points,
                image_points=image_points,
                camera_matrix=camera_matrix,
                base_to_camera_optical=base_to_camera_optical,
            )
            if perturbed_residual_vector is None or not perturbed_valid_depth:
                jacobian_columns.append(np.zeros_like(weighted_residual_vector))
                continue
            jacobian_columns.append((perturbed_residual_vector - residual_vector) / step_eps[parameter_index])

        jacobian = np.column_stack(jacobian_columns) * sqrt_weights[:, None]
        normal_matrix = jacobian.T @ jacobian + np.eye(3, dtype=np.float64) * lambda_diag
        gradient = jacobian.T @ weighted_residual_vector
        try:
            delta = -np.linalg.solve(normal_matrix, gradient)
        except np.linalg.LinAlgError:
            break

        candidate_params = best_params + delta
        candidate_residual_vector, candidate_residual_norms, candidate_valid_depth = _residuals_from_planar_pose(
            params=candidate_params,
            world_points=world_points,
            image_points=image_points,
            camera_matrix=camera_matrix,
            base_to_camera_optical=base_to_camera_optical,
        )
        if candidate_residual_vector is None or not candidate_valid_depth:
            lambda_diag *= 10.0
            continue

        candidate_rmse = float(np.sqrt(np.mean(candidate_residual_norms * candidate_residual_norms)))
        if candidate_rmse + 1e-9 < best_rmse:
            best_params = candidate_params
            best_rmse = candidate_rmse
            lambda_diag = max(lambda_diag * 0.5, 1e-6)
            if float(np.linalg.norm(delta)) < 1e-5:
                break
        else:
            lambda_diag *= 10.0

    translation_update_norm = float(np.linalg.norm(best_params[:2] - params[:2]))
    yaw_update_deg = float(np.degrees(best_params[2] - params[2]))
    refined_planar_pose = PlanarPose(
        x=float(best_params[0]),
        y=float(best_params[1]),
        yaw=float(best_params[2]),
    )
    refined_residual_vector, refined_residual_norms, refined_valid_depth = _residuals_from_planar_pose(
        params=best_params,
        world_points=world_points,
        image_points=image_points,
        camera_matrix=camera_matrix,
        base_to_camera_optical=base_to_camera_optical,
    )
    if refined_residual_vector is None or not refined_valid_depth:
        return RefineResult(
            attempted=True,
            accepted=False,
            status="invalid_refined_projection",
            refined_planar_pose=current_planar_pose,
            vis_lm_count=int(matched_track_ids.size),
            grid_cells=int(grid_cell_count),
            rmse_before=rmse_before,
            rmse_after=float("inf"),
            matched_track_ids=matched_track_ids,
            resid_px=np.empty((0,), dtype=np.float64),
        )

    if not best_rmse + 1e-9 < rmse_before:
        status = "rmse_not_improved"
        accepted = False
    elif abs(yaw_update_deg) > config.max_yaw_deg:
        status = "yaw_update_too_large"
        accepted = False
    elif translation_update_norm > config.max_step_ratio * max(current_step_length_m, 1e-3):
        status = "translation_update_too_large"
        accepted = False
    else:
        status = "refinement_applied"
        accepted = True

    return RefineResult(
        attempted=True,
        accepted=accepted,
        status=status,
        refined_planar_pose=refined_planar_pose if accepted else current_planar_pose,
        vis_lm_count=int(matched_track_ids.size),
        grid_cells=int(grid_cell_count),
        rmse_before=rmse_before,
        rmse_after=best_rmse,
        matched_track_ids=matched_track_ids,
        resid_px=refined_residual_norms if accepted else np.empty((0,), dtype=np.float64),
    )


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous_points = np.hstack(
        [np.asarray(points, dtype=np.float64), np.ones((points.shape[0], 1), dtype=np.float64)]
    )
    return (transform @ homogeneous_points.T).T[:, :3]


def _project_world_points(
    world_points: np.ndarray,
    world_from_camera_optical: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_from_world = invert_transform(world_from_camera_optical)
    camera_points = _transform_points(camera_from_world, world_points)
    depths = camera_points[:, 2]
    valid_depth_mask = depths > _MIN_DEPTH_M
    image_points = np.full((world_points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid_depth_mask):
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        valid_points = camera_points[valid_depth_mask]
        image_points[valid_depth_mask, 0] = fx * valid_points[:, 0] / valid_points[:, 2] + cx
        image_points[valid_depth_mask, 1] = fy * valid_points[:, 1] / valid_points[:, 2] + cy
    return image_points, depths, valid_depth_mask


def _mean_two_view_reprojection_errors(
    world_points: np.ndarray,
    previous_keyframe: KeyframeRec,
    current_keyframe: KeyframeRec,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    previous_projected, _, previous_valid = _project_world_points(
        world_points=world_points,
        world_from_camera_optical=previous_keyframe.world_from_camera_optical,
        camera_matrix=camera_matrix,
    )
    current_projected, _, current_valid = _project_world_points(
        world_points=world_points,
        world_from_camera_optical=current_keyframe.world_from_camera_optical,
        camera_matrix=camera_matrix,
    )
    valid_mask = previous_valid & current_valid
    mean_errors = np.full((world_points.shape[0],), np.inf, dtype=np.float64)
    if np.any(valid_mask):
        previous_errors = np.linalg.norm(previous_projected[valid_mask] - previous_points[valid_mask], axis=1)
        current_errors = np.linalg.norm(current_projected[valid_mask] - current_points[valid_mask], axis=1)
        mean_errors[valid_mask] = 0.5 * (previous_errors + current_errors)
    return mean_errors


def _residuals_from_planar_pose(
    params: np.ndarray,
    world_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    base_to_camera_optical: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    planar_pose = PlanarPose(x=float(params[0]), y=float(params[1]), yaw=float(params[2]))
    world_from_base = world_from_planar_base_pose(planar_pose)
    world_from_camera_optical = world_from_base @ base_to_camera_optical
    projected_points, depths, valid_depth_mask = _project_world_points(
        world_points=world_points,
        world_from_camera_optical=world_from_camera_optical,
        camera_matrix=camera_matrix,
    )
    if not np.all(valid_depth_mask):
        return None, np.empty((0,), dtype=np.float64), False
    residuals_2d = projected_points - image_points
    residual_norms = np.linalg.norm(residuals_2d, axis=1)
    return residuals_2d.reshape(-1), residual_norms, bool(np.all(depths > _MIN_DEPTH_M))


def _huber_weights(residual_norms: np.ndarray, delta_px: float) -> np.ndarray:
    weights = np.ones_like(residual_norms, dtype=np.float64)
    if delta_px <= 0.0:
        return weights
    large_mask = residual_norms > float(delta_px)
    weights[large_mask] = float(delta_px) / residual_norms[large_mask]
    return weights
