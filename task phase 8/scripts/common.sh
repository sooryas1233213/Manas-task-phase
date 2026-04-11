#!/usr/bin/env bash

TASK8_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="task8"
TASK8_URL="http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale"

docker_compose() {
  docker compose -f "${TASK8_DIR}/compose.yaml" "$@"
}

container_id() {
  docker_compose ps -q "${SERVICE_NAME}"
}

wait_for_health() {
  local timeout_secs="${1:-180}"
  local container=""
  local started_at=""
  local state=""
  local health=""

  container="$(container_id)"
  if [[ -z "${container}" ]]; then
    echo "Task 8 container was not created."
    return 1
  fi

  started_at="$(date +%s)"

  while true; do
    read -r state health < <(
      docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container}"
    )

    if [[ "${state}" == "running" && ( "${health}" == "healthy" || "${health}" == "none" ) ]]; then
      return 0
    fi

    if [[ "${state}" == "exited" || "${state}" == "dead" || "${health}" == "unhealthy" ]]; then
      echo "Task 8 container failed to become healthy."
      docker_compose logs --tail 120 "${SERVICE_NAME}" || true
      return 1
    fi

    if (( $(date +%s) - started_at >= timeout_secs )); then
      echo "Timed out waiting for Task 8 container health after ${timeout_secs}s."
      docker_compose logs --tail 120 "${SERVICE_NAME}" || true
      return 1
    fi

    sleep 2
  done
}

start_stack() {
  local timeout_secs="${1:-180}"
  local rebuild="${2:-0}"

  if [[ "${rebuild}" == "1" ]]; then
    docker_compose up --build -d
  else
    docker_compose up -d
  fi

  wait_for_health "${timeout_secs}"
}
