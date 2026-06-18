#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/run_demo.sh [options] [-- extra_isaac_args...]

Starts the full AeroStrike ROS 2 policy runtime and Isaac bridge from one terminal.

Options:
  --headless                 Run Isaac without the viewport.
  --stop-on-termination      Exit after the first terminal episode.
  --steps N                  Isaac env steps to run. Default: 4500 (~90s).
  --layout-seed N            Warehouse layout seed. Default: 7.
  --real-time-factor X       Isaac wall-clock pacing. Default: 1.0.
  --ros-startup-wait X       Seconds to let ROS launch settle. Default: 4.
  --build                    Build aerostrike_pkg before launching.
  -h, --help                 Show this help.

Examples:
  ./scripts/run_demo.sh
  ./scripts/run_demo.sh --stop-on-termination
  ./scripts/run_demo.sh --headless --steps 2500
EOF
}

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
steps=4500
layout_seed=7
real_time_factor=1.0
ros_startup_wait_s=4
visible=true
stop_on_termination=false
build_first=false
extra_isaac_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless)
            visible=false
            shift
            ;;
        --stop-on-termination)
            stop_on_termination=true
            shift
            ;;
        --steps)
            steps="$2"
            shift 2
            ;;
        --layout-seed)
            layout_seed="$2"
            shift 2
            ;;
        --real-time-factor)
            real_time_factor="$2"
            shift 2
            ;;
        --ros-startup-wait)
            ros_startup_wait_s="$2"
            shift 2
            ;;
        --build)
            build_first=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra_isaac_args+=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

ros_launch_pid=""
cleaned_up=false

cleanup() {
    if [[ "${cleaned_up}" == "true" ]]; then
        return
    fi
    cleaned_up=true

    if [[ -n "${ros_launch_pid}" ]] && kill -0 "${ros_launch_pid}" 2>/dev/null; then
        echo
        echo "[demo] Stopping ROS runtime..."
        if kill -0 "-${ros_launch_pid}" 2>/dev/null; then
            kill -INT "-${ros_launch_pid}" 2>/dev/null || true
            for _ in {1..20}; do
                if ! kill -0 "-${ros_launch_pid}" 2>/dev/null; then
                    wait "${ros_launch_pid}" 2>/dev/null || true
                    return
                fi
                sleep 0.5
            done
            echo "[demo] ROS runtime did not stop after SIGINT; sending SIGTERM..."
            kill -TERM "-${ros_launch_pid}" 2>/dev/null || true
            sleep 2
            if kill -0 "-${ros_launch_pid}" 2>/dev/null; then
                echo "[demo] ROS runtime did not stop after SIGTERM; sending SIGKILL..."
                kill -KILL "-${ros_launch_pid}" 2>/dev/null || true
            fi
        else
            kill -INT "${ros_launch_pid}" 2>/dev/null || true
        fi
        wait "${ros_launch_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

cd "${workspace_root}"

if [[ "${build_first}" == "true" ]] || [[ ! -f "${workspace_root}/install/setup.bash" ]]; then
    echo "[demo] Building aerostrike_pkg..."
    colcon build --packages-select aerostrike_pkg --cmake-force-configure
fi

set +u
# shellcheck source=/dev/null
source "${workspace_root}/setup_env.sh"
set -u

isaaclab_sh="${ISAACLAB_SH:-}"
if [[ -z "${isaaclab_sh}" ]]; then
    if [[ -n "${ISAACLAB_PATH:-}" ]]; then
        isaaclab_sh="${ISAACLAB_PATH}/isaaclab.sh"
    else
        isaaclab_sh="/home/user/IsaacLab/isaaclab.sh"
    fi
fi
if [[ ! -x "${isaaclab_sh}" ]]; then
    echo "Isaac Lab launcher not found or not executable: ${isaaclab_sh}" >&2
    echo "Set ISAACLAB_SH=/path/to/isaaclab.sh or ISAACLAB_PATH=/path/to/IsaacLab." >&2
    exit 1
fi

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
    export TERM="xterm-256color"
fi

isaac_ros_bridge="${ISAAC_ROS_BRIDGE:-/home/user/isaacsim-5.1.0/exts/isaacsim.ros2.bridge/humble}"
if [[ ! -d "${isaac_ros_bridge}" ]]; then
    echo "Isaac ROS bridge path not found: ${isaac_ros_bridge}" >&2
    echo "Set ISAAC_ROS_BRIDGE=/path/to/isaacsim.ros2.bridge/humble." >&2
    exit 1
fi

echo "[demo] Starting ROS runtime..."
setsid ros2 launch aerostrike_pkg policy_runtime.launch.xml &
ros_launch_pid=$!

sleep "${ros_startup_wait_s}"
if ! kill -0 "${ros_launch_pid}" 2>/dev/null; then
    echo "ROS launch exited before Isaac started." >&2
    wait "${ros_launch_pid}" || true
    exit 1
fi

export ISAAC_ROS_BRIDGE="${isaac_ros_bridge}"
export PYTHONPATH="${ISAAC_ROS_BRIDGE}/rclpy:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ISAAC_ROS_BRIDGE}/lib:${LD_LIBRARY_PATH:-}"

isaac_args=(
    -p "${workspace_root}/scripts/run_ros_runtime_sim.py"
    --layout_seed "${layout_seed}"
    --steps "${steps}"
    --real_time_factor "${real_time_factor}"
)

if [[ "${visible}" == "true" ]]; then
    isaac_args+=(--visible)
fi
if [[ "${stop_on_termination}" == "true" ]]; then
    isaac_args+=(--stop_on_termination)
fi
isaac_args+=("${extra_isaac_args[@]}")

echo "[demo] Starting Isaac bridge..."
echo "[demo] Metrics CSV: ${workspace_root}/logs/ros_metrics/latest.csv"
set +e
"${isaaclab_sh}" "${isaac_args[@]}"
isaac_status=$?
set -e

cleanup

if [[ -f "${workspace_root}/logs/ros_metrics/latest.csv" ]]; then
    echo
    echo "[demo] Final metrics row:"
    tail -n 1 "${workspace_root}/logs/ros_metrics/latest.csv"
fi

exit "${isaac_status}"
