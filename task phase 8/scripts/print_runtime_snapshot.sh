#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

container="$(container_id)"
if [[ -z "${container}" ]]; then
  echo "Task 8 container is not running."
  exit 1
fi

docker_compose exec -T "${SERVICE_NAME}" bash -lc '
CV_BRIDGE_VERSION="$(dpkg-query -W -f='"'"'${Version}'"'"' ros-humble-cv-bridge 2>/dev/null || echo unknown)"
python3 - <<'"'"'PY'"'"'
import cv2
import hashlib
from pathlib import Path

video_path = Path("/project/video.mp4")
sha256 = hashlib.sha256()
with video_path.open("rb") as video_file:
    for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
        sha256.update(chunk)

print(f"OpenCV {cv2.__version__}")
print(f"video_sha256 {sha256.hexdigest()}")
PY
echo "cv_bridge ${CV_BRIDGE_VERSION}"
'
