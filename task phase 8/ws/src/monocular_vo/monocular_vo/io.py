from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from sensor_msgs.msg import CameraInfo


@dataclass
class CameraProcessingState:
    camera_info: CameraInfo
    camera_matrix: np.ndarray
    distortion: np.ndarray
    map1: Optional[np.ndarray]
    map2: Optional[np.ndarray]


def build_approx_camera_info(
    width: int,
    height: int,
    horizontal_fov_deg: float,
    frame_id: str,
) -> CameraInfo:
    if width <= 0 or height <= 0:
        raise ValueError("CameraInfo dimensions must be positive.")
    if horizontal_fov_deg <= 0.0 or horizontal_fov_deg >= 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180 degrees.")

    fx = width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [
        float(fx), 0.0, float(cx),
        0.0, float(fy), float(cy),
        0.0, 0.0, 1.0,
    ]
    msg.r = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    msg.p = [
        float(fx), 0.0, float(cx), 0.0,
        0.0, float(fy), float(cy), 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    return msg


def load_camera_info_from_yaml(
    yaml_path: str,
    output_width: int,
    output_height: int,
    frame_id: str,
) -> CameraInfo:
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera calibration YAML not found: {yaml_path}")

    data = yaml.safe_load(path.read_text()) or {}
    if "camera_matrix" not in data:
        raise ValueError("Calibration YAML must contain camera_matrix.")

    input_width = int(data.get("image_width", output_width))
    input_height = int(data.get("image_height", output_height))
    if input_width <= 0 or input_height <= 0:
        raise ValueError("Calibration YAML contains invalid image dimensions.")

    sx = output_width / float(input_width)
    sy = output_height / float(input_height)

    k = [float(v) for v in data["camera_matrix"]["data"]]
    if len(k) != 9:
        raise ValueError("camera_matrix.data must have 9 elements.")
    k[0] *= sx
    k[2] *= sx
    k[4] *= sy
    k[5] *= sy

    d = [float(v) for v in data.get("distortion_coefficients", {}).get("data", [0.0] * 5)]
    r = [float(v) for v in data.get("rectification_matrix", {}).get("data", [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ])]
    projection_data = data.get("projection_matrix", {}).get("data")
    if projection_data is None:
        p = [
            k[0], 0.0, k[2], 0.0,
            0.0, k[4], k[5], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
    else:
        p = [float(v) for v in projection_data]
        p[0] *= sx
        p[2] *= sx
        p[5] *= sy
        p[6] *= sy
    if len(p) != 12:
        raise ValueError("projection_matrix.data must have 12 elements.")

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = output_width
    msg.height = output_height
    msg.distortion_model = data.get("distortion_model", "plumb_bob")
    msg.d = d
    msg.k = k
    msg.r = r
    msg.p = p
    return msg


def build_camera_processing_state(camera_info: CameraInfo) -> CameraProcessingState:
    camera_matrix = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
    distortion = np.array(camera_info.d, dtype=np.float64).reshape(-1, 1)
    map1 = None
    map2 = None

    if distortion.size > 0 and np.any(np.abs(distortion) > 1e-12):
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            np.eye(3, dtype=np.float64),
            camera_matrix,
            (int(camera_info.width), int(camera_info.height)),
            cv2.CV_16SC2,
        )

    return CameraProcessingState(
        camera_info=camera_info,
        camera_matrix=camera_matrix,
        distortion=distortion,
        map1=map1,
        map2=map2,
    )


def preprocess_frame(
    image_bgr: np.ndarray,
    processing_state: CameraProcessingState,
    use_rectified_images: bool,
    enable_clahe: bool,
) -> tuple[np.ndarray, np.ndarray]:
    processed = image_bgr

    if not use_rectified_images and processing_state.map1 is not None and processing_state.map2 is not None:
        processed = cv2.remap(image_bgr, processing_state.map1, processing_state.map2, cv2.INTER_LINEAR)

    grayscale = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if enable_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grayscale = clahe.apply(grayscale)

    return grayscale, processed
