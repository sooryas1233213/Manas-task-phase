# Task Phase 8: ROS 2 Humble + RViz in Docker on Apple Silicon

This task provides a native `arm64` ROS 2 Humble desktop workflow for an M2 Pro MacBook Pro using Docker Desktop and a browser-based noVNC desktop. RViz runs inside the Linux container and is exposed locally at `http://127.0.0.1:6080`.

## What Is Included

- `compose.yaml`: single-service Docker Compose stack for ROS 2 Humble + RViz + noVNC
- `Dockerfile`: builds on `arm64v8/ros:humble-ros-base-jammy` and installs RViz plus desktop services
- `run_task8.sh`: builds and starts the stack, waits for health, launches RViz, and opens the browser desktop
- `smoke_test.sh`: launches a static TF and marker publisher, then verifies the RViz smoke-test topics
- `shell_task8.sh`: opens an interactive ROS-sourced shell in the container
- `build_ws.sh`: builds packages under `task phase 8/ws/src` if any exist
- `stop_task8.sh`: shuts the stack down cleanly
- `ws/`: bind-mounted ROS workspace for any packages you want to add later

## Prerequisites

- Docker Desktop running on macOS
- Apple Silicon host (`arm64` / M-series)
- Local browser access to `127.0.0.1:6080`

## First Start

```bash
cd /Users/sooryas/Code/Manas\ task\ phase/task\ phase\ 8
./run_task8.sh
```

What happens:

- Docker builds the native `arm64` image
- Compose starts `Xvfb`, `fluxbox`, `x11vnc`, and `websockify`
- the script waits until the noVNC web desktop is healthy
- RViz launches inside the container with software rendering enabled
- the browser desktop opens at `http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale`

## Smoke Test

```bash
cd /Users/sooryas/Code/Manas\ task\ phase/task\ phase\ 8
./smoke_test.sh
```

The smoke test:

- starts a static transform from `map` to `marker_frame`
- publishes a green sphere marker on `/visualization_marker`
- makes sure `/tf_static` and `/visualization_marker` appear
- ensures `rviz2` is running with the task 8 RViz config

In RViz, you should see:

- `Fixed Frame` set to `map`
- `Grid`
- `TF`
- `Marker`

## Workspace Usage

Put packages under:

```bash
/Users/sooryas/Code/Manas task phase/task phase 8/ws/src
```

Build them with:

```bash
cd /Users/sooryas/Code/Manas\ task\ phase/task\ phase\ 8
./build_ws.sh
```

Open a ROS-sourced shell:

```bash
cd /Users/sooryas/Code/Manas\ task\ phase/task\ phase\ 8
./shell_task8.sh
```

## Stop

```bash
cd /Users/sooryas/Code/Manas\ task\ phase/task\ phase\ 8
./stop_task8.sh
```

## Notes

- The noVNC port is bound only to `127.0.0.1`, so the web desktop stays local to this Mac.
- RViz uses software rendering by default for stability on macOS Docker.
- If you add workspace packages and rebuild, new shells automatically source `/ws/install/setup.bash` when it exists.
- Container-side logs are written under `/tmp/task8-*.log` and supervisor logs live under `/var/log/supervisor/`.
