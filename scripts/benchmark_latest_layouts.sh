#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-16}"
EPISODES="${EPISODES:-32}"
CHECKPOINT="${CHECKPOINT:-/home/user/drone_ws/logs/skrl}"
LAYOUT_SEEDS=(7 11 19 23)

for layout_seed in "${LAYOUT_SEEDS[@]}"; do
  echo "[INFO] Benchmarking latest checkpoint layout_seed=${layout_seed}"
  /home/user/IsaacLab/isaaclab.sh -p /home/user/drone_ws/scripts/benchmark.py \
    --headless \
    --num_envs "${NUM_ENVS}" \
    --episodes "${EPISODES}" \
    --checkpoint "${CHECKPOINT}" \
    --layout_seed="${layout_seed}" \
    "$@"
done
