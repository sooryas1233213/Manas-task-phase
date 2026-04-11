from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackingConfig:
    max_features: int
    min_tracked_points: int
    min_feature_distance_px: float
    shi_tomasi_quality_level: float
    shi_tomasi_block_size: int
    lk_win_size: int
    lk_max_level: int
    lk_max_iterations: int
    lk_epsilon: float
    feature_border_px: int
    klt_fb_max_error_px: float
    orb_max_features: int
    grid_rows: int
    grid_cols: int


@dataclass(frozen=True)
class TrackingHealth:
    total_input_points: int
    forward_survivor_count: int
    fb_survivor_count: int
    median_fb_error_px: float


@dataclass(frozen=True)
class TrackResult:
    previous_points: np.ndarray
    current_points: np.ndarray
    survivor_indices: np.ndarray
    rejected_previous_points: np.ndarray
    rejected_current_points: np.ndarray
    health: TrackingHealth


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _merge_points(*point_sets: np.ndarray) -> np.ndarray:
    valid_sets = [points.reshape(-1, 2).astype(np.float32) for points in point_sets if points.size > 0]
    if not valid_sets:
        return _empty_points()
    return np.vstack(valid_sets)


def _cell_edges(size: int, count: int) -> np.ndarray:
    count = max(1, int(count))
    return np.linspace(0, size, count + 1, dtype=int)


def _cell_quotas(total: int, num_cells: int) -> list[int]:
    if total <= 0 or num_cells <= 0:
        return [0] * max(0, num_cells)
    base = total // num_cells
    remainder = total % num_cells
    return [base + (1 if index < remainder else 0) for index in range(num_cells)]


def _build_detection_mask(
    image_gray: np.ndarray,
    config: TrackingConfig,
    existing_points: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.full(image_gray.shape[:2], 255, dtype=np.uint8)
    border = max(0, int(config.feature_border_px))
    if border > 0 and image_gray.shape[0] > border * 2 and image_gray.shape[1] > border * 2:
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0

    if existing_points is not None and existing_points.size > 0:
        suppression_radius = max(1, int(round(config.min_feature_distance_px)))
        for point in existing_points.reshape(-1, 2):
            x_coord, y_coord = np.round(point).astype(int)
            cv2.circle(mask, (x_coord, y_coord), suppression_radius, 0, thickness=-1)

    return mask


def detect_features(
    image_gray: np.ndarray,
    config: TrackingConfig,
    existing_points: np.ndarray | None = None,
    max_features: int | None = None,
) -> np.ndarray:
    max_features = int(config.max_features if max_features is None else max_features)
    if max_features <= 0:
        return _empty_points()

    base_mask = _build_detection_mask(image_gray, config, existing_points=existing_points)
    y_edges = _cell_edges(image_gray.shape[0], config.grid_rows)
    x_edges = _cell_edges(image_gray.shape[1], config.grid_cols)
    quotas = _cell_quotas(max_features, config.grid_rows * config.grid_cols)

    detections: list[np.ndarray] = []
    cell_index = 0
    for row_index in range(config.grid_rows):
        for col_index in range(config.grid_cols):
            quota = quotas[cell_index]
            cell_index += 1
            if quota <= 0:
                continue

            y_start, y_end = int(y_edges[row_index]), int(y_edges[row_index + 1])
            x_start, x_end = int(x_edges[col_index]), int(x_edges[col_index + 1])
            if y_end <= y_start or x_end <= x_start:
                continue

            cell_mask = base_mask[y_start:y_end, x_start:x_end]
            if cell_mask.size == 0 or np.count_nonzero(cell_mask) == 0:
                continue

            cell_image = image_gray[y_start:y_end, x_start:x_end]
            cell_features = cv2.goodFeaturesToTrack(
                cell_image,
                maxCorners=quota,
                qualityLevel=float(config.shi_tomasi_quality_level),
                minDistance=float(config.min_feature_distance_px),
                blockSize=int(config.shi_tomasi_block_size),
                mask=cell_mask,
                useHarrisDetector=False,
            )
            if cell_features is None:
                continue

            offset = np.array([x_start, y_start], dtype=np.float32)
            detections.append(cell_features.reshape(-1, 2).astype(np.float32) + offset)

    return _merge_points(*detections)


def detect_orb_features(
    image_gray: np.ndarray,
    config: TrackingConfig,
    existing_points: np.ndarray | None = None,
    max_features: int | None = None,
) -> np.ndarray:
    max_features = int(config.orb_max_features if max_features is None else max_features)
    if max_features <= 0:
        return _empty_points()

    base_mask = _build_detection_mask(image_gray, config, existing_points=existing_points)
    detector = cv2.ORB_create(
        nfeatures=max(max_features * 3, config.orb_max_features),
        edgeThreshold=max(31, int(config.feature_border_px)),
    )
    keypoints = detector.detect(image_gray, mask=base_mask)
    if not keypoints:
        return _empty_points()

    quotas = _cell_quotas(max_features, config.grid_rows * config.grid_cols)
    selected_points: list[np.ndarray] = []
    cell_counts = [0] * len(quotas)
    width = max(1, image_gray.shape[1])
    height = max(1, image_gray.shape[0])

    for keypoint in sorted(keypoints, key=lambda kp: kp.response, reverse=True):
        col_index = min(config.grid_cols - 1, int(keypoint.pt[0] * config.grid_cols / width))
        row_index = min(config.grid_rows - 1, int(keypoint.pt[1] * config.grid_rows / height))
        cell_index = row_index * config.grid_cols + col_index
        if quotas[cell_index] <= 0 or cell_counts[cell_index] >= quotas[cell_index]:
            continue

        selected_points.append(np.array(keypoint.pt, dtype=np.float32))
        cell_counts[cell_index] += 1
        if len(selected_points) >= max_features:
            break

    if not selected_points:
        return _empty_points()
    return np.stack(selected_points).astype(np.float32)


def track_features(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: np.ndarray,
    config: TrackingConfig,
) -> TrackResult:
    if previous_points.size == 0:
        return TrackResult(
            previous_points=_empty_points(),
            current_points=_empty_points(),
            survivor_indices=np.empty((0,), dtype=np.int64),
            rejected_previous_points=_empty_points(),
            rejected_current_points=_empty_points(),
            health=TrackingHealth(
                total_input_points=0,
                forward_survivor_count=0,
                fb_survivor_count=0,
                median_fb_error_px=float("inf"),
            ),
        )

    previous_points_2d = previous_points.reshape(-1, 2).astype(np.float32)
    lk_previous = previous_points_2d.reshape(-1, 1, 2)
    forward_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        lk_previous,
        None,
        winSize=(int(config.lk_win_size), int(config.lk_win_size)),
        maxLevel=int(config.lk_max_level),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(config.lk_max_iterations),
            float(config.lk_epsilon),
        ),
    )

    if forward_points is None or forward_status is None:
        return TrackResult(
            previous_points=_empty_points(),
            current_points=_empty_points(),
            survivor_indices=np.empty((0,), dtype=np.int64),
            rejected_previous_points=previous_points_2d,
            rejected_current_points=previous_points_2d.copy(),
            health=TrackingHealth(
                total_input_points=int(previous_points_2d.shape[0]),
                forward_survivor_count=0,
                fb_survivor_count=0,
                median_fb_error_px=float("inf"),
            ),
        )

    forward_points_2d = forward_points.reshape(-1, 2).astype(np.float32)
    safe_forward_points = np.where(np.isfinite(forward_points_2d), forward_points_2d, previous_points_2d)
    forward_status_mask = forward_status.reshape(-1).astype(bool)
    forward_finite_mask = np.isfinite(previous_points_2d).all(axis=1) & np.isfinite(safe_forward_points).all(axis=1)
    forward_keep_mask = forward_status_mask & forward_finite_mask

    border = float(max(0, int(config.feature_border_px)))
    width = float(current_gray.shape[1])
    height = float(current_gray.shape[0])
    previous_in_bounds = (
        (previous_points_2d[:, 0] >= border)
        & (previous_points_2d[:, 0] < width - border)
        & (previous_points_2d[:, 1] >= border)
        & (previous_points_2d[:, 1] < height - border)
    )
    current_in_bounds = (
        (safe_forward_points[:, 0] >= border)
        & (safe_forward_points[:, 0] < width - border)
        & (safe_forward_points[:, 1] >= border)
        & (safe_forward_points[:, 1] < height - border)
    )
    forward_keep_mask &= previous_in_bounds & current_in_bounds

    forward_previous_points = previous_points_2d[forward_keep_mask]
    forward_current_points = safe_forward_points[forward_keep_mask]
    forward_source_indices = np.nonzero(forward_keep_mask)[0].astype(np.int64)
    forward_survivor_count = int(forward_previous_points.shape[0])

    if forward_survivor_count == 0:
        return TrackResult(
            previous_points=_empty_points(),
            current_points=_empty_points(),
            survivor_indices=np.empty((0,), dtype=np.int64),
            rejected_previous_points=previous_points_2d,
            rejected_current_points=safe_forward_points,
            health=TrackingHealth(
                total_input_points=int(previous_points_2d.shape[0]),
                forward_survivor_count=0,
                fb_survivor_count=0,
                median_fb_error_px=float("inf"),
            ),
        )

    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        forward_current_points.reshape(-1, 1, 2),
        None,
        winSize=(int(config.lk_win_size), int(config.lk_win_size)),
        maxLevel=int(config.lk_max_level),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(config.lk_max_iterations),
            float(config.lk_epsilon),
        ),
    )

    if backward_points is None or backward_status is None:
        return TrackResult(
            previous_points=_empty_points(),
            current_points=_empty_points(),
            rejected_previous_points=previous_points_2d,
            rejected_current_points=safe_forward_points,
            health=TrackingHealth(
                total_input_points=int(previous_points_2d.shape[0]),
                forward_survivor_count=forward_survivor_count,
                fb_survivor_count=0,
                median_fb_error_px=float("inf"),
            ),
        )

    backward_points_2d = backward_points.reshape(-1, 2).astype(np.float32)
    backward_status_mask = backward_status.reshape(-1).astype(bool)
    backward_finite_mask = np.isfinite(backward_points_2d).all(axis=1)
    fb_errors = np.linalg.norm(backward_points_2d - forward_previous_points, axis=1)
    fb_keep_mask = backward_status_mask & backward_finite_mask & (fb_errors <= float(config.klt_fb_max_error_px))

    rejected_previous_points = _merge_points(previous_points_2d[~forward_keep_mask], forward_previous_points[~fb_keep_mask])
    rejected_current_points = _merge_points(safe_forward_points[~forward_keep_mask], forward_current_points[~fb_keep_mask])
    survivor_indices = forward_source_indices[fb_keep_mask]
    valid_fb_errors = fb_errors[backward_status_mask & backward_finite_mask]

    return TrackResult(
        previous_points=forward_previous_points[fb_keep_mask],
        current_points=forward_current_points[fb_keep_mask],
        survivor_indices=survivor_indices,
        rejected_previous_points=rejected_previous_points,
        rejected_current_points=rejected_current_points,
        health=TrackingHealth(
            total_input_points=int(previous_points_2d.shape[0]),
            forward_survivor_count=forward_survivor_count,
            fb_survivor_count=int(fb_keep_mask.sum()),
            median_fb_error_px=float(np.median(valid_fb_errors)) if valid_fb_errors.size > 0 else float("inf"),
        ),
    )
