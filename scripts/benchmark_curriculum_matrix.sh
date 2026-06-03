#!/usr/bin/env bash
set -euo pipefail

CHECKPOINTS=(
  "baseline:/home/user/drone_ws/logs/skrl/aerostrike_navigation/2026-05-25_18-53-15_ppo_torch/checkpoints/best_agent.pt"
  "seed11:/home/user/drone_ws/logs/skrl/aerostrike_navigation/2026-06-03_20-34-54_ppo_torch/checkpoints/best_agent.pt"
  "seed23:/home/user/drone_ws/logs/skrl/aerostrike_navigation/2026-06-03_20-42-06_ppo_torch/checkpoints/best_agent.pt"
  "seed19:/home/user/drone_ws/logs/skrl/aerostrike_navigation/2026-06-03_20-46-27_ppo_torch/checkpoints/best_agent.pt"
)

LAYOUT_SEEDS=(7 11 19 23)

for checkpoint_entry in "${CHECKPOINTS[@]}"; do
  checkpoint_name="${checkpoint_entry%%:*}"
  checkpoint_path="${checkpoint_entry#*:}"

  for layout_seed in "${LAYOUT_SEEDS[@]}"; do
    echo "[INFO] Benchmarking checkpoint=${checkpoint_name} layout_seed=${layout_seed}"
    /home/user/IsaacLab/isaaclab.sh -p /home/user/drone_ws/scripts/benchmark.py \
      --headless \
      --num_envs 16 \
      --episodes 32 \
      --checkpoint "${checkpoint_path}" \
      --layout_seed="${layout_seed}" \
      "$@"
  done
done
