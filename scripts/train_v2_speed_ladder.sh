#!/usr/bin/env bash
set -Eeuo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${workspace_root}"

task_id="AeroStrike-Navigation-V2-Direct-v0"
num_envs="${AEROSTRIKE_V2_NUM_ENVS:-8}"
max_iterations="${AEROSTRIKE_V2_MAX_ITERATIONS:-}"

args=(
    scripts/train.py
    --task "${task_id}"
    --num_envs "${num_envs}"
)

if [[ -n "${max_iterations}" ]]; then
    args+=(--max_iterations "${max_iterations}")
fi

exec "${ISAACLAB_SH:-/home/user/IsaacLab/isaaclab.sh}" -p "${args[@]}" "$@"
