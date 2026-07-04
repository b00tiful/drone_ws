#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./scripts/export_policy_v2.sh CHECKPOINT [extra export_policy.py args...]" >&2
    exit 2
fi

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="$1"
shift

cd "${workspace_root}"

exec "${ISAACLAB_SH:-/home/user/IsaacLab/isaaclab.sh}" -p scripts/export_policy.py \
    --checkpoint "${checkpoint}" \
    --output "${workspace_root}/checkpoints/aerostrike_policy_v2.onnx" \
    --metadata "${workspace_root}/checkpoints/aerostrike_policy_v2.yaml" \
    --navigation-config "${workspace_root}/configs/navigation_v2.yaml" \
    --quadrotor-config "${workspace_root}/configs/quadrotor_v2.yaml" \
    "$@"
