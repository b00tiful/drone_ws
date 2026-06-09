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
export ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-${AEROSTRIKE_WS}/.deps/onnxruntime-linux-x64-1.26.0}"

if [ -d "${ONNXRUNTIME_ROOT}/lib" ]; then
    export LD_LIBRARY_PATH="${ONNXRUNTIME_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${ROS_LOG_DIR:-}" ] && [ ! -w "${HOME}/.ros/log" ]; then
    export ROS_LOG_DIR="/tmp/aerostrike_ros_logs"
    mkdir -p "${ROS_LOG_DIR}"
fi

if [ -n "${ISAACLAB_PATH}" ] && [ -f "${ISAACLAB_PATH}/isaaclab.sh" ]; then
    # shellcheck source=/dev/null
    source "${ISAACLAB_PATH}/isaaclab.sh"
fi

if [ -f "/opt/ros/humble/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "/opt/ros/humble/setup.bash"
fi

if [ -f "${AEROSTRIKE_WS}/install/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "${AEROSTRIKE_WS}/install/setup.bash"
fi

echo "AeroStrike workspace: ${AEROSTRIKE_WS}"
