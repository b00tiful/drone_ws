#!/usr/bin/env bash
set -euo pipefail

exec /home/user/IsaacLab/isaaclab.sh -p /home/user/drone_ws/scripts/train.py \
  --headless \
  --num_envs 16 \
  --max_iterations 500 \
  --checkpoint /home/user/drone_ws/logs/skrl \
  --layout_seed=23 \
  "$@"
