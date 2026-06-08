#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-16}"
CYCLES="${CYCLES:-3}"
ITERATIONS_PER_LAYOUT="${ITERATIONS_PER_LAYOUT:-100}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/user/drone_ws/logs/skrl}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/user/drone_ws/logs/skrl/aerostrike_navigation/2026-05-25_18-53-15_ppo_torch/checkpoints/best_agent.pt}"
LAYOUT_SEEDS=(7 11 23 19)

checkpoint="${BASE_CHECKPOINT}"

for ((cycle = 1; cycle <= CYCLES; cycle++)); do
  for layout_seed in "${LAYOUT_SEEDS[@]}"; do
    echo "[INFO] Interleaved training cycle=${cycle}/${CYCLES} layout_seed=${layout_seed} checkpoint=${checkpoint}"
    /home/user/IsaacLab/isaaclab.sh -p /home/user/drone_ws/scripts/train.py \
      --headless \
      --num_envs "${NUM_ENVS}" \
      --max_iterations "${ITERATIONS_PER_LAYOUT}" \
      --checkpoint "${checkpoint}" \
      --layout_seed="${layout_seed}"

    checkpoint="${CHECKPOINT_ROOT}"
  done
done
