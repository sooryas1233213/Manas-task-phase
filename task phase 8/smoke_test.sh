#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/common.sh"

START_TIMEOUT="${TASK8_START_TIMEOUT:-240}"

echo "Ensuring the Task 8 container is running..."
start_stack "${START_TIMEOUT}"

echo "Starting the RViz smoke test demo..."
docker_compose exec -T "${SERVICE_NAME}" bash -lc "/opt/task8/scripts/smoke_demo.sh"

echo
echo "Smoke test passed."
echo "Open the browser desktop at:"
echo "${TASK8_URL}"

if [[ "${OPEN_BROWSER:-1}" == "1" ]] && command -v open >/dev/null 2>&1; then
  open "${TASK8_URL}" >/dev/null 2>&1 || true
fi
