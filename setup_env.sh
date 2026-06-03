#!/usr/bin/env bash

# AeroStrike environment bootstrap.
# Source this file from the workspace root:
#   source setup_env.sh

export AEROSTRIKE_WS="${AEROSTRIKE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONPATH="${AEROSTRIKE_WS}/aerostrike_lab:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Set these locally if your Isaac Sim / Isaac Lab installs live elsewhere.
export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-}"

if [ -n "${ISAACLAB_PATH}" ] && [ -f "${ISAACLAB_PATH}/isaaclab.sh" ]; then
    # shellcheck source=/dev/null
    source "${ISAACLAB_PATH}/isaaclab.sh"
fi

if [ -f "/opt/ros/humble/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "/opt/ros/humble/setup.bash"
fi

echo "AeroStrike workspace: ${AEROSTRIKE_WS}"
