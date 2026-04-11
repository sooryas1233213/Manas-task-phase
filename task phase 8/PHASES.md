# Task 8 Dashcam Monocular VO Phases

This document is the implementation roadmap for the dashcam-only monocular visual odometry task that will run inside the existing Task 8 ROS 2 Humble + RViz environment.

It does not replace [`README.md`](/Users/sooryas/Code/Manas%20task%20phase/task%20phase%208/README.md). The existing README stays focused on Docker, ROS 2, RViz, and workspace usage. This file defines how the visual odometry system should be built in phases so that each stage works on its own and then integrates cleanly into the final system.

## Problem Statement

The car starts at `(0, 0)` with heading `0 deg` and only has a forward-facing dash camera. The system must estimate positional change in `x` and `y` plus orientation `yaw`, publish the result as `nav_msgs/Odometry`, and visualize the growing path in RViz as frames progress.

The core approach is:

- start with a simple but correct monocular VO baseline
- keep the ROS interfaces fixed from the beginning
- add robustness before complexity
- make metric scale a required final capability
- integrate mapping and optimization only after the baseline path is stable

## Task 8 Assumptions

- Task 8 already provides a working ROS 2 Humble container, RViz desktop, and workspace build flow.
- All new code will live under [`task phase 8/ws/src`](/Users/sooryas/Code/Manas%20task%20phase/task%20phase%208/ws/src).
- The implementation will use Python ROS 2 (`rclpy`) plus OpenCV for the first complete system.
- Metric scale is required in the final integrated pipeline, with camera height plus ground-plane recovery as the default strategy.
- Loop closure and learned depth are optional extensions, not part of the required first complete system.

## Target Package Layout

Create one ROS 2 package named `monocular_vo` under [`task phase 8/ws/src`](/Users/sooryas/Code/Manas%20task%20phase/task%20phase%208/ws/src).

```text
task phase 8/ws/src/monocular_vo/
├── package.xml
├── setup.py
├── resource/monocular_vo
├── launch/
│   └── vo.launch.py
├── config/
│   └── vo.yaml
└── monocular_vo/
    ├── __init__.py
    ├── vo_node.py
    ├── io.py
    ├── frontend.py
    ├── geometry.py
    ├── pose_integration.py
    ├── scale.py
    ├── mapping.py
    ├── optimization.py
    └── debug_viz.py
```

Recommended responsibilities:

- `vo_node.py`: ROS node, parameter loading, publishers, subscriptions, TF, and message assembly
- `io.py`: image conversion, CameraInfo handling, undistortion maps, preprocessing
- `frontend.py`: Shi-Tomasi, KLT tracking, forward-backward filtering, ORB reseeding, feature grid logic
- `geometry.py`: Essential matrix, homography checks, `recoverPose`, triangulation, PnP helpers
- `pose_integration.py`: world pose state, yaw extraction, quaternion generation, path buffering
- `scale.py`: ground-point selection, plane fit, camera-height-based scale estimation and smoothing
- `mapping.py`: keyframes, landmarks, local map state, track management
- `optimization.py`: local bundle adjustment in a sliding window
- `debug_viz.py`: debug overlays, counters, and health metrics

## ROS Interface Contract

These interfaces should stay stable across all phases.

### Inputs

- `/camera/image_raw` as `sensor_msgs/Image`
- `/camera/camera_info` as `sensor_msgs/CameraInfo`

If input comes from a recorded video instead of a live ROS stream, use a separate bridge or replay node that republishes frames onto the same topics. The VO node should not depend on a separate internal input mode.

### Outputs

- `/vo/odom` as `nav_msgs/Odometry`
- `/vo/path` as `nav_msgs/Path`
- `odom -> base_link` as dynamic TF
- `/vo/debug_tracks` as an optional debug image topic

### Frames

Use these frames as the single source of truth:

- `odom`
- `base_link`
- `camera_link`
- `camera_optical_frame`

VO can be computed in `camera_optical_frame`, but the published odometry must be consistent with `odom` and `base_link`.

### Timing Rules

- Odometry, Path, and TF must use the image timestamp from the incoming frame.
- Do not stamp outputs with ad hoc wall-clock time if the frame already has a valid acquisition time.
- If a frame is rejected by health gates, skip the pose update instead of publishing a fabricated motion step.

### Minimum Parameters

Expose these parameters in `config/vo.yaml` from the start:

- `camera_height_m`
- `use_rectified_images`
- `max_features`
- `min_inliers`
- `min_parallax_px`
- `orb_reseed_threshold`
- `scale_mode`

Use `scale_mode: ground_plane` as the default final-system mode, even if early phases temporarily operate with placeholder scale inside the same integration pipeline.

## Phase 0: Package, Data Flow, and RViz Wiring

### Goal

Establish a stable ROS package and data path so every later algorithm plugs into the same node graph and topic layout.

### What Gets Added

- create the `monocular_vo` package inside the Task 8 workspace
- add a launch file that starts the VO node with a single config file
- subscribe to `/camera/image_raw` and `/camera/camera_info`
- cache camera intrinsics and build undistortion maps from `CameraInfo`
- add grayscale conversion and mild normalization such as CLAHE
- support rectified and non-rectified input via config
- publish an initialized but stationary `/vo/odom` and `/vo/path`
- publish static frame configuration needed for RViz sanity
- add a replay path for rosbag or a separate video-to-ROS publisher that uses the same camera topics

### Exit Criteria

- `./build_ws.sh` succeeds after the package is added
- the node starts inside Task 8 without modifying container orchestration
- image and `CameraInfo` callbacks are live
- RViz can display `odom`, TF, and an empty or stationary path without frame errors
- preprocessing and undistortion can be toggled through config rather than code edits

## Phase 1: Baseline Monocular VO

### Goal

Produce the first working path using classic two-view monocular geometry with the smallest reliable algorithm stack.

### What Gets Added

- detect Shi-Tomasi corners on the first valid frame
- track points with pyramidal Lucas-Kanade optical flow
- estimate the Essential matrix with RANSAC
- recover relative pose with `recoverPose`
- accumulate orientation and translation direction into a global pose state
- extract yaw from the world rotation and publish a planar `x`, `y`, `yaw` odometry output
- publish quaternions derived from yaw with `z = 0`
- append each accepted pose to `/vo/path`
- keep translation scale plumbed through a dedicated interface, using a temporary placeholder scale factor while the scale module is not yet integrated

### Exit Criteria

- a short forward-driving clip produces a continuous path in RViz
- heading changes are visible and consistent with obvious turns
- the TF tree remains valid while the car moves
- no hard-coded topic names or timestamps need to change in later phases

## Phase 2: Robustness and Failure Gating

### Goal

Make the baseline survive real dashcam conditions without corrupting the trajectory when image quality or scene geometry becomes unreliable.

### What Gets Added

- forward-backward KLT consistency checks to remove unstable tracks
- minimum tracked-point and minimum inlier thresholds before pose acceptance
- median epipolar error and feature-count health metrics
- parallax gating so translation updates are frozen when motion is near rotation-only
- yaw-only updates when translation is ill-conditioned but rotation remains usable
- ORB-based reseeding when KLT tracks fall below threshold
- grid-based feature distribution to avoid point clustering on moving cars or a single image region
- conservative handling of planar degeneracy, including optional Essential-vs-Homography comparison when correspondence geometry suggests a homography-dominant frame
- debug overlay output showing tracks, inliers, and rejection reasons

### Exit Criteria

- blurred or low-texture segments no longer cause obvious pose jumps
- rotation-heavy clips update heading without exploding translation
- dynamic objects do not dominate the path estimate when static background support is still present
- rejected frames are skipped cleanly and do not break downstream RViz visualization

## Phase 3: Metric Scale Integration

### Goal

Turn the arbitrary-scale motion pipeline into a metric-ish trajectory using only dashcam-compatible priors, with camera height plus ground-plane estimation as the default strategy.

### What Gets Added

- maintain short multi-frame feature tracks suitable for triangulation
- select candidate ground-region points, favoring the lower image region and stable track direction
- triangulate candidate 3D points from accepted frame pairs or local keyframe pairs
- fit a ground plane robustly from candidate 3D points
- estimate camera-to-plane distance in VO units
- compute scale using `camera_height_m / estimated_vo_height`
- low-pass filter scale estimates to avoid visible scale jitter
- apply the filtered scale before translation is integrated into the world pose
- keep a confidence measure so low-quality scale estimates reuse the previous stable scale rather than injecting noise

### Exit Criteria

- the same pipeline now produces translation in plausible metric units
- route length over a known segment is within a reasonable tolerance for a monocular student system
- scale remains stable across several consecutive frames instead of oscillating frame by frame
- when ground-plane estimation is weak, the system holds the last good scale instead of resetting the path

## Phase 4: Keyframes and Local Mapping

### Goal

Reduce drift by introducing a small local map without changing the external ROS interface.

### What Gets Added

- keyframe creation based on rotation threshold, feature loss, or sufficient parallax
- a sliding local map of roughly 5 to 10 keyframes
- triangulated landmarks anchored to keyframes
- persistent landmark tracks across multiple frames
- PnP with RANSAC to estimate the current pose against local landmarks
- fallback to Essential-matrix frame-to-frame pose only when local map support is insufficient
- clean separation between front-end tracking and map-backed pose estimation

### Exit Criteria

- the path drifts less on longer clips than the Essential-only baseline
- pose estimation remains stable even when direct frame-to-frame matches temporarily degrade
- local mapping integrates without changing topics, frame names, or launch structure

## Phase 5: Local Bundle Adjustment and Final Integration

### Goal

Refine the local map and recent poses through bounded optimization while preserving the same external behavior that earlier phases established.

### What Gets Added

- sliding-window bundle adjustment over recent keyframes and landmarks
- reprojection-error residuals with robust loss
- optimizer hooks in `optimization.py` so the rest of the system is not tied to one solver implementation
- a default Python-first local optimizer path, for example `scipy.optimize.least_squares`, to stay aligned with the Python ROS 2 stack
- window maintenance logic so optimization remains bounded and predictable
- post-optimization pose/state updates feeding the same `Odometry`, `Path`, and TF publishers already used in earlier phases

### Exit Criteria

- recent trajectory segments are smoother and more self-consistent than the Phase 4 output
- optimization failures do not crash the node or publish invalid poses
- the final integrated system still uses the same launch file, topics, and frame contract as Phase 0

## Phase 6: Demo Readiness, Evaluation, and Bonus Work

### Goal

Make the system reliable to demonstrate, easy to debug, and clear about what is required versus optional.

### What Gets Added

- a final RViz configuration for odometry, path, TF, and optional debug imagery
- logging for frame health, inlier count, parallax, scale confidence, and optimization status
- a debugging checklist covering bad intrinsics, missing `CameraInfo`, frame mismatches, timestamp mistakes, and degenerate motion
- repeatable test clips for forward motion, turning, blur, low texture, and dynamic traffic
- optional loop closure as a bonus only after the full scaled VO pipeline is stable
- optional place-recognition and pose-graph work only if the required phases are already working end to end

### Exit Criteria

- a complete demo can be run fully inside the Task 8 ROS and RViz environment
- RViz shows a stable `odom`-framed path and valid TF tree
- the system has a documented checklist for diagnosing common VO failures
- optional loop closure is clearly separated from the required deliverable

## Integration Rules

- Do not create throwaway prototype nodes that bypass the final ROS interfaces.
- Keep the same package name, launch file, topics, and frames from Phase 0 onward.
- Every new subsystem must plug into the same `monocular_vo` node graph rather than replacing it with a parallel stack.
- If a frame is low confidence, skip the update or fall back conservatively. Never publish fabricated high-confidence motion.
- Additive debug outputs are allowed, but downstream consumers must rely only on the stable public interfaces.
- Each phase must preserve the previous RViz demo instead of breaking it and promising later cleanup.
- Feature tracking, scale, mapping, and optimization should remain modular internally, but integration must stay stable externally.

## Validation and Test Plan

### Per-Phase Smoke Test

- package builds with [`build_ws.sh`](/Users/sooryas/Code/Manas%20task%20phase/task%20phase%208/build_ws.sh)
- Task 8 container and RViz start with [`run_task8.sh`](/Users/sooryas/Code/Manas%20task%20phase/task%20phase%208/run_task8.sh)
- the node launches and expected topics appear
- RViz displays TF and the path topic without frame mismatch warnings

### Baseline VO Test

- use a short forward-driving clip
- verify smooth odometry updates and a growing path
- confirm there are no large discontinuities or teleports

### Turning Test

- use a clip with a visible turn or near-rotation-only segment
- confirm yaw updates remain stable
- confirm translation is reduced or frozen when parallax is too small

### Robustness Test

- use clips with blur, dynamic traffic, glare, and low texture
- verify bad frames are rejected or downweighted
- verify reseeding restores tracking instead of restarting the system

### Scale Test

- compare estimated traveled distance against a known approximate route length or a manually measured segment
- verify scale smoothing removes visible jitter
- confirm bad scale estimates do not reset the trajectory

### Integration Test

- run the entire system inside the existing Task 8 container flow
- confirm no changes are required to the Docker orchestration for later phases
- confirm the same launch and topic structure is preserved across phases

## Final Acceptance Checklist

- `nav_msgs/Odometry` is published continuously on `/vo/odom`
- `nav_msgs/Path` is published continuously on `/vo/path`
- `odom -> base_link` TF is valid in RViz
- path and odometry use image timestamps
- the final system estimates `x`, `y`, and `yaw`
- metric scale is integrated through camera height plus ground-plane estimation
- optional loop closure is documented as bonus work, not required work

## Recommended Build and Run Flow

Once the package exists, keep the workflow aligned with Task 8:

```bash
cd /Users/sooryas/Code/Manas\ task\ phase/task\ phase\ 8
./run_task8.sh
./build_ws.sh
./shell_task8.sh
```

From the container shell, the intended final launch shape is:

```bash
source /opt/task8/scripts/task8_ros.sh
ros2 launch monocular_vo vo.launch.py
```

This keeps the implementation inside the environment that Task 8 already standardized.
