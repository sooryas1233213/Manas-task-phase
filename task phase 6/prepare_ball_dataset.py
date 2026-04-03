from __future__ import annotations

import argparse
import importlib.util
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CandidateSample:
    # One possible training example for the ball detector.
    frame_index: int
    center: tuple[int, int]
    radius: float
    score: float
    labeled: bool


def load_v2_module(module_path: Path):
    # Reuse the older tracker as a trainer to auto-generate rough ball labels.
    spec = importlib.util.spec_from_file_location("volleyball_tracker_v2_bootstrap", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tracker module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    # This script creates the training dataset for the custom ball detector.
    #
    # Final output:
    # datasets/ball/
    # datasets/ball_dataset.yaml
    #
    # The workflow is:
    # 1. Run the older tracker on the video
    # 2. Collect confident ball frames as positives
    # 3. Collect some ball-missing frames as negatives
    # 4. Save everything in YOLO dataset format
    phase_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare a YOLO dataset for active volleyball detection.")
    parser.add_argument(
        "--input",
        type=Path,
        default=phase_dir / "Volleyball.mp4",
        help="Input volleyball video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=phase_dir / "datasets" / "ball",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=phase_dir / "volleyball_tracker_v2.py",
        help="Teacher tracker used for pseudo-label bootstrapping.",
    )
    parser.add_argument(
        "--max-positive",
        type=int,
        default=520,
        help="Maximum number of positive labeled frames to export.",
    )
    parser.add_argument(
        "--max-negative",
        type=int,
        default=120,
        help="Maximum number of negative frames to export.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for train/val split.",
    )
    return parser.parse_args()


def reset_dataset_root(dataset_root: Path) -> None:
    # Start from a clean YOLO dataset folder every time.
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_masks(module, config, shape: tuple[int, int]) -> np.ndarray:
    return module.apply_exclusions(
        module.build_mask(shape, config.ball_search_polygon),
        config.ball_exclusion_rects,
    )


def to_yolo_line(
    center: tuple[int, int],
    radius: float,
    frame_width: int,
    frame_height: int,
) -> str:
    # Convert a ball center/radius into YOLO label format:
    # class_id center_x center_y width height
    box_size = max(10.0, radius * 2.7)
    width_norm = min(box_size / frame_width, 1.0)
    height_norm = min(box_size / frame_height, 1.0)
    x_norm = center[0] / frame_width
    y_norm = center[1] / frame_height
    return f"0 {x_norm:.6f} {y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"


def sample_score(
    frame_index: int,
    center: tuple[int, int],
    radius: float,
    last_center: Optional[tuple[int, int]],
    frame_width: int,
    frame_height: int,
) -> float:
    # Prefer harder ball examples:
    # near edges, near the top, moving fast, or very small.
    edge_margin = min(center[0], frame_width - center[0], center[1], frame_height - center[1])
    edge_bonus = 1.0 if edge_margin < 150 else 0.0
    top_bonus = 0.8 if center[1] < 180 else 0.0
    speed_bonus = 0.0
    if last_center is not None:
        speed_bonus = min(np.hypot(center[0] - last_center[0], center[1] - last_center[1]) / 50.0, 2.0)
    radius_bonus = 0.6 if radius < 6.0 else 0.0
    temporal_bonus = (frame_index % 47) / 47.0
    return edge_bonus + top_bonus + speed_bonus + radius_bonus + temporal_bonus


def collect_teacher_samples(args: argparse.Namespace) -> tuple[list[CandidateSample], list[int], tuple[int, int]]:
    # Run the old tracker over the video and collect:
    # 1. positive frames where the ball looks confidently tracked
    # 2. negative frames where the ball seems missing
    module = load_v2_module(args.teacher)
    config = module.TrackerConfig()

    capture = cv2.VideoCapture(str(args.input), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {args.input}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    search_mask = build_masks(module, config, (frame_height, frame_width))
    tracker = module.BallTrackerV2(config, search_mask)

    previous_gray: Optional[np.ndarray] = None
    positives: list[CandidateSample] = []
    negatives: list[int] = []
    last_confirmed_center: Optional[tuple[int, int]] = None
    missing_streak = 0

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        # Build the same helper inputs the old tracker expects.
        # - blurred frame
        # - HSV frame
        # - motion mask
        blurred = cv2.GaussianBlur(frame, config.blur_kernel, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        if previous_gray is None:
            motion_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        else:
            frame_delta = cv2.absdiff(gray, previous_gray)
            _, motion_mask = cv2.threshold(
                frame_delta,
                config.ball_motion_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            motion_mask = cv2.medianBlur(motion_mask, 3)
        previous_gray = gray

        sample = tracker.update(frame, hsv, motion_mask)
        if sample.status == "confirmed" and sample.center is not None and sample.radius is not None:
            # Save confident ball positions as future training labels.
            score = sample_score(
                frame_index,
                sample.center,
                sample.radius,
                last_confirmed_center,
                frame_width,
                frame_height,
            )
            positives.append(
                CandidateSample(
                    frame_index=frame_index,
                    center=sample.center,
                    radius=sample.radius,
                    score=score,
                    labeled=True,
                )
            )
            last_confirmed_center = sample.center
            missing_streak = 0
        else:
            # If the teacher cannot find the ball for a while, store a few empty frames too.
            missing_streak += 1
            if missing_streak >= 12 and frame_index % 24 == 0:
                negatives.append(frame_index)

        frame_index += 1

    capture.release()
    return positives, negatives, (frame_width, frame_height)


def pick_export_indices(
    positives: list[CandidateSample],
    negatives: list[int],
    max_positive: int,
    max_negative: int,
) -> tuple[dict[int, CandidateSample], set[int]]:
    # Keep a mix of:
    # 1. hard/high-value ball examples
    # 2. evenly spread examples across the video
    if not positives:
        raise RuntimeError("Teacher tracker produced no positive ball samples.")

    ranked = sorted(positives, key=lambda item: item.score, reverse=True)
    hard_quota = min(len(ranked), max(80, max_positive // 3))
    chosen: dict[int, CandidateSample] = {item.frame_index: item for item in ranked[:hard_quota]}

    remaining_slots = max_positive - len(chosen)
    if remaining_slots > 0:
        uniform = sorted(positives, key=lambda item: item.frame_index)
        stride = max(1, len(uniform) // max(remaining_slots, 1))
        for item in uniform[::stride]:
            chosen.setdefault(item.frame_index, item)
            if len(chosen) >= max_positive:
                break

    chosen_negative = set(sorted(set(negatives))[:: max(1, len(set(negatives)) // max(max_negative, 1))][:max_negative])
    return chosen, chosen_negative


def export_dataset(
    args: argparse.Namespace,
    positive_samples: dict[int, CandidateSample],
    negative_samples: set[int],
    frame_size: tuple[int, int],
) -> None:
    # Write images and labels in the folder structure YOLO expects.
    #
    # YOLO wants:
    # images/train
    # images/val
    # labels/train
    # labels/val
    #
    # Each image gets a .txt file with the same base name.
    # Positive images get one "ball" box.
    # Negative images get an empty label file.
    dataset_root = args.output
    reset_dataset_root(dataset_root)

    rng = random.Random(args.seed)
    positive_indices = list(positive_samples.keys())
    negative_indices = list(negative_samples)
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)

    val_positive = set(positive_indices[: int(len(positive_indices) * args.val_ratio)])
    val_negative = set(negative_indices[: int(len(negative_indices) * args.val_ratio)])

    frame_width, frame_height = frame_size
    capture = cv2.VideoCapture(str(args.input), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to reopen video: {args.input}")

    wanted_frames = set(positive_indices) | set(negative_indices)
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index not in wanted_frames:
            frame_index += 1
            continue

        split = "val" if frame_index in val_positive or frame_index in val_negative else "train"
        image_path = dataset_root / "images" / split / f"frame_{frame_index:05d}.jpg"
        label_path = dataset_root / "labels" / split / f"frame_{frame_index:05d}.txt"
        cv2.imwrite(str(image_path), frame)

        if frame_index in positive_samples:
            # Positive frame: write one ball label.
            candidate = positive_samples[frame_index]
            label_path.write_text(
                to_yolo_line(candidate.center, candidate.radius, frame_width, frame_height),
                encoding="utf-8",
            )
        else:
            # Negative frame: write an empty label file.
            label_path.write_text("", encoding="utf-8")

        frame_index += 1

    capture.release()


def write_dataset_yaml(dataset_root: Path) -> None:
    # YOLO uses this YAML file to know where the train/val folders are.
    yaml_path = dataset_root.parent / "ball_dataset.yaml"
    yaml_path.write_text(
        (
            f"path: {dataset_root.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: ball\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    # Step 1:
    # Run the teacher tracker over the full video.
    # This gives us candidate positive and negative examples.
    positives, negatives, frame_size = collect_teacher_samples(args)

    # Step 2:
    # Pick a manageable subset of those frames.
    # We keep hard examples and also spread examples across time.
    chosen_positives, chosen_negatives = pick_export_indices(
        positives,
        negatives,
        args.max_positive,
        args.max_negative,
    )

    # Step 3:
    # Save the actual image files and YOLO label files.
    export_dataset(args, chosen_positives, chosen_negatives, frame_size)

    # Step 4:
    # Write the dataset YAML so YOLO training knows where train/val data lives.
    write_dataset_yaml(args.output)
    print(
        {
            "positive_samples": len(chosen_positives),
            "negative_samples": len(chosen_negatives),
            "dataset_root": str(args.output),
            "dataset_yaml": str(args.output.parent / "ball_dataset.yaml"),
        }
    )


if __name__ == "__main__":
    main()
