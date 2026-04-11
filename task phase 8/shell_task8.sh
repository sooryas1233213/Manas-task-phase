#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/common.sh"

START_TIMEOUT="${TASK8_START_TIMEOUT:-240}"

start_stack "${START_TIMEOUT}"
exec docker_compose exec "${SERVICE_NAME}" bash -l
