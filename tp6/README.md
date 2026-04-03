# TP6 OpenCV-First Task 6

This folder contains OpenCV-heavy Task 6 implementations. The newest primary path is `volleyball_tracker_opencv_v9.py`, which keeps the stable v5-v8 player pipeline unchanged and makes the ball tracker more liberal again in safe central regions while keeping the exact hard false-positive zones blocked.

## Versions

- `volleyball_tracker_opencv.py`
  - baseline OpenCV-only v1
- `volleyball_tracker_opencv_v2.py`
  - stronger v2 with:
    - confirmation-based ball states (`SEARCH_INIT`, `TRACK_CONTOUR`, `TRACK_BRIDGE`, `SEARCH_REACQUIRE`)
    - motion-seeded ball locking to avoid banner and torso false positives
    - sparse detector correction for ball reacquisition when the OpenCV contour search drifts
    - top/bottom player pipelines with separate foreground cleanup
    - short-term CamShift player maintenance plus zone-based count stabilization
- `volleyball_tracker_opencv_v3.py`
  - strict OpenCV-only v3 with:
    - offline 2-pass pipeline
    - motion-first ball observations with Kalman track states
    - short prediction / optical-flow coast mode instead of random ball guesses
    - running-average background extraction for players
    - per-player Kalman tracks with zone-based count stabilization
    - HSV torso classification kept only for team labeling
- `volleyball_tracker_opencv_v4.py`
  - clean ground-up strict OpenCV v4 with:
    - offline 2-pass `BackgroundSubtractorMOG2` observations for both ball and players
    - motion-first ball candidates plus short Kalman coast/recover states
    - no optical flow, no CamShift, no detector fallback
    - classical centroid / footpoint player tracking with zone-based count smoothing
    - HSV torso voting only for team labels
- `volleyball_tracker_opencv_v5.py`
  - ball-first trajectory rebuild with:
    - offline observation pass with approximate-median motion + `BackgroundSubtractorMOG2`
    - trajectory-grown ball segments instead of per-frame contour locking
    - constant-acceleration Kalman smoothing and short-gap interpolation
    - recovery-only use of `task phase 6/models/ball_best.pt` after real long-loss windows
    - v4-style classical player tracking kept as the stability baseline
- `volleyball_tracker_opencv_v6.py`
  - ball-only rebuild on top of the stable v5 player path with:
    - a 4D constant-velocity Kalman tracker and ROI-first ball states (`SEARCH_INIT`, `TRACK`, `DEGRADED`, `RECOVER`, `LOST`)
    - label-derived ball size profiling from `task phase 6/datasets/ball`
    - OpenCV candidate sieves plus trajectory residual checks and short-gap interpolation
    - optional recovery-only `RandomForestClassifier` cached at `tp6/_artifacts/ball_recovery_rf.joblib`
    - no YOLO or Ultralytics in the runtime script
- `volleyball_tracker_opencv_v7.py`
  - offscreen-safe ball tracker on top of the unchanged v6 player path with:
    - explicit `OFFSCREEN` state so high/out-of-frame exits go blank instead of coasting into false anchors
    - safe re-entry masking that excludes the top banner and sideline-official risk strips during startup/recovery
    - stricter interpolation/coast rules so trails stop at offscreen gaps instead of scribbling through them
    - recovery-only `RandomForestClassifier` cached at `tp6/_artifacts/ball_recovery_rf_v7.joblib`
- `volleyball_tracker_opencv_v8.py`
  - balanced ball tracker on top of the unchanged v7 player path with:
    - hard bans only for the exact left-head, right-head, and top-banner false-positive zones
    - soft edge restrictions used only during offscreen recovery, not global high-ball re-entry
    - short `OFFSCREEN_GRACE` bridging before full offscreen blanking
    - recovery-only `RandomForestClassifier` cached at `tp6/_artifacts/ball_recovery_rf_v8.joblib`
- `volleyball_tracker_opencv_v9.py`
  - liberal reacquisition ball tracker on top of the unchanged v8 player path with:
    - the same hard bans for the exact left-head, right-head, and top-banner false-positive zones
    - earlier handoff from high-ball loss into recovery instead of long offscreen blanking
    - more central high-ball candidate retention and lower safe-region anchor floors
    - recovery-only `RandomForestClassifier` cached at `tp6/_artifacts/ball_recovery_rf_v9.joblib`

## Pipeline

- `volleyball_tracker_opencv_v3.py`:
  - Pass 1 caches raw OpenCV observations:
    - blurred grayscale motion masks for the ball
    - running-average foreground blobs for players
  - Pass 2 replays the video and applies:
    - ball tracking from motion candidates + Kalman + short coast/recover states
    - player tracking from foreground blobs + Kalman association + zone holds
    - HSV torso color voting for stable team labels

- `volleyball_tracker_opencv_v4.py`:
  - Pass 1 caches raw OpenCV observations:
    - `BackgroundSubtractorMOG2` ball blobs inside the playable-air mask
    - separate top / bottom player foreground blobs
  - Pass 2 replays the video and applies:
    - ball tracking from temporal small-blob support + Kalman predict/correct
    - player tracking from contour blobs + centroid association
    - HSV torso color voting for stable team labels

- `volleyball_tracker_opencv_v5.py`:
  - Pass 1 caches trajectory-friendly observations:
    - approximate-median motion blobs and `BackgroundSubtractorMOG2` ball blobs
    - separate top / bottom player foreground blobs
  - Pass 2 builds player states:
    - classical centroid / footpoint tracking
    - HSV torso color voting for stable team labels
  - Pass 3 builds ball segments:
    - candidate sieves
    - Kalman-guided trajectory growth
    - linear / quadratic plausibility checks
    - short-gap interpolation
    - sparse recovery-only ball detector usage when long gaps remain

- `volleyball_tracker_opencv_v6.py`:
  - Pass 1 caches the same OpenCV observations as v5 for players and ball candidates
  - Pass 2 keeps the v5 player tracker unchanged
  - Pass 3 rebuilds the ball path with:
    - y-aware ball size gates from the labeled dataset
    - Kalman ROI tracking with degraded and recover states
    - offline interpolation plus jump-pruning
    - optional RandomForest recovery re-ranking only during real loss windows

- `volleyball_tracker_opencv_v7.py`:
  - Pass 1 caches the same OpenCV observations as v6
  - Pass 2 keeps the v6 player tracker unchanged
  - Pass 3 hardens only the ball path with:
    - explicit offscreen transitions and conservative blanking at high exits
    - safe re-entry gating and clip-specific sideline/top-banner risk suppression
    - trail resets across offscreen/lost jumps
    - richer debug CSV fields for `mode`, `x/y`, `offscreen`, `risk_strip`, and `reentry_reject`

- `volleyball_tracker_opencv_v8.py`:
  - Pass 1 caches the same OpenCV observations as v7
  - Pass 2 keeps the v7 player tracker unchanged
  - Pass 3 rebalances only the ball path with:
    - hard-risk suppression for the exact known false-positive rectangles
    - central high-ball re-entry restored outside those rectangles
    - `OFFSCREEN_GRACE` short bridging before full offscreen blanking
    - richer debug CSV fields for `risk_zone` and `offscreen_grace`

- `volleyball_tracker_opencv_v9.py`:
  - Pass 1 caches the same OpenCV observations as v8
  - Pass 2 keeps the v8 player tracker unchanged
  - Pass 3 liberalizes only the ball path with:
    - more per-frame central candidates kept alive into scoring
    - earlier recovery after high arcs instead of waiting in offscreen blanking
    - stronger central reacquire bonuses outside torso suppression and hard-risk rectangles
    - richer debug CSV fields for `central_reacquire`

## Run

```bash
python tp6/volleyball_tracker_opencv.py
python tp6/volleyball_tracker_opencv_v2.py
python tp6/volleyball_tracker_opencv_v3.py
python tp6/volleyball_tracker_opencv_v4.py
python tp6/volleyball_tracker_opencv_v5.py
python tp6/volleyball_tracker_opencv_v6.py
python tp6/volleyball_tracker_opencv_v7.py
python tp6/volleyball_tracker_opencv_v8.py
python tp6/volleyball_tracker_opencv_v9.py
```

Optional flags:

```bash
python tp6/volleyball_tracker_opencv.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv.mp4" --display
python tp6/volleyball_tracker_opencv_v2.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv_v2.mp4" --display
python tp6/volleyball_tracker_opencv_v3.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv_v3.mp4" --display
python tp6/volleyball_tracker_opencv_v4.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv_v4.mp4" --display
python tp6/volleyball_tracker_opencv_v5.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv_v5.mp4" --debug-dir "tp6/_v5_check"
python tp6/volleyball_tracker_opencv_v6.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv_v6.mp4" --debug-dir "tp6/_v6_check"
python tp6/volleyball_tracker_opencv_v7.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv_v7.mp4" --debug-dir "tp6/_v7_check"
python tp6/volleyball_tracker_opencv_v8.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv_v8.mp4" --debug-dir "tp6/_v8_check"
python tp6/volleyball_tracker_opencv_v9.py --input "task phase 6/Volleyball.mp4" --output "tp6/Volleyball_annotated_opencv_v9.mp4" --debug-dir "tp6/_v9_check"
```

## Notes

- These implementations are clip-specific and use fixed court/air masks.
- `volleyball_tracker_opencv_v9.py` is the new primary path for this clip.
- V9 keeps the player side identical to v8 and changes only the ball tracker.
- V9 uses no YOLO or Ultralytics at runtime; the optional recovery classifier is trained from the existing labeled ball dataset.
- In the current render, v9 materially improves continuity over v8: `833` rendered ball states vs `570`, `432` confirmed vs `269`, and `58` offscreen frames vs `231`, while keeping hard-risk confirmed/interpolated states at `0`.
- Mild undercount during heavy player overlap is still possible because the scene is single-camera and player detection remains detector-free.
