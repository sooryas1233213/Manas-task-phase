from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    # This script trains the custom YOLO model that detects only the volleyball.
    #
    # The output of this script is:
    # models/ball_best.pt
    #
    # That file is what the final runtime tracker loads.
    phase_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a clip-specific YOLO detector for the volleyball.")
    parser.add_argument(
        "--data",
        type=Path,
        default=phase_dir / "datasets" / "ball_dataset.yaml",
        help="Dataset YAML for ball detection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Starting YOLO model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Training device.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=phase_dir / "runs" / "ball_train",
        help="Training project directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="volleyball_ball",
        help="Training run name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=phase_dir / "models" / "ball_best.pt",
        help="Destination for the trained best weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    # Step 1:
    # Load a pretrained YOLO model.
    #
    # We do NOT train from scratch.
    # We start from yolov8s.pt so the model already understands general image features.
    # Then we fine-tune it to learn one specific class: the volleyball.
    # Start from a pretrained YOLO model instead of training from scratch.
    # This is standard transfer learning and works better for small datasets.
    model = YOLO(args.model)

    # Step 2:
    # Train the model on our custom dataset.
    #
    # Important points:
    # - data=... points to ball_dataset.yaml
    # - single_cls=True means only one class exists: "ball"
    # - imgsz is high because the volleyball is tiny in the video
    # - device="mps" uses Apple Metal on this machine
    # - augmentation is mild because this is a clip-specific model
    # Train a one-class detector that only learns "ball".
    # The dataset comes from prepare_ball_dataset.py.
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        single_cls=True,
        cache=True,
        pretrained=True,
        close_mosaic=5,
        degrees=0.0,
        scale=0.15,
        translate=0.05,
        fliplr=0.0,
        mosaic=0.2,
    )

    # Step 3:
    # Training creates a run folder with checkpoints.
    # Ultralytics saves the best checkpoint as:
    # runs/.../weights/best.pt
    #
    # We copy that file into models/ball_best.pt so the runtime script
    # always knows where to find the final trained detector.
    # After training, copy the best checkpoint to a stable path
    # so the runtime tracker can load it directly.
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training completed but best weights are missing: {best_weights}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.output)
    print({"best_weights": str(args.output), "save_dir": str(results.save_dir)})


if __name__ == "__main__":
    main()
