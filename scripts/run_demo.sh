#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/run_demo.sh [options] [-- extra_isaac_args...]

Starts the full AeroStrike ROS 2 policy runtime and Isaac bridge from one terminal.

Options:
  --headless                 Run Isaac without the viewport.
  --profile NAME             Runtime profile: v1 or v2. Default: v1.
  --stop-on-termination      Exit after the first terminal episode.
  --steps N                  Isaac env steps to run. Default: 4500 (~90s).
  --layout-seed N            Warehouse layout seed. Default: 7.
  --scene-variant NAME       Scene variant: warehouse, hallway, or long_warehouse. Default: warehouse.
  --camera-mode MODE         Visible camera: third_person, first_person, or free. Default: third_person.
  --camera-distance X        Follow camera distance in meters. Default: 0.40.
  --camera-height X          Follow camera height above drone root. Default: 0.25.
  --camera-target-height X   Camera look-at height above drone root. Default: 0.20.
  --camera-smoothing X       Follow camera smoothing alpha [0, 1]. Default: 0.50.
  --camera-max-yaw-rate X    Follow camera heading limit in deg/s. Default: 90.
  --demo-robot-marker        Show a non-physics visual marker on the drone.
  --demo-robot-marker-radius X
                             Marker axis half-length in meters. Default: 0.16.
  --real-time-factor X       Isaac wall-clock pacing. Default: 0.0.
  --command-timeout X        Seconds before stale commands are zeroed. Default: 1.0.
  --ros-startup-wait X       Seconds to let ROS launch settle. Default: 4.
  --build                    Build aerostrike_pkg before launching.
  -h, --help                 Show this help.

Examples:
  ./scripts/run_demo.sh
  ./scripts/run_demo.sh --scene-variant hallway --stop-on-termination
  ./scripts/run_demo.sh --camera-mode first_person
  ./scripts/run_demo.sh --stop-on-termination
  ./scripts/run_demo.sh --headless --steps 2500
  ./scripts/run_demo.sh --profile v2 --headless --stop-on-termination --steps 6000
EOF
}

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile=v1
steps=4500
layout_seed=7
scene_variant=warehouse
camera_mode=third_person
camera_distance_m=0.40
camera_height_m=0.25
camera_target_height_m=0.20
camera_smoothing=0.50
camera_max_yaw_rate_dps=90.0
demo_robot_marker=false
demo_robot_marker_radius_m=0.16
real_time_factor=0.0
command_timeout_s=1.0
ros_startup_wait_s=4
visible=true
stop_on_termination=false
build_first=false
extra_isaac_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            profile="$2"
            shift 2
            ;;
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
        --scene-variant)
            scene_variant="$2"
            shift 2
            ;;
        --camera-mode)
            camera_mode="$2"
            shift 2
            ;;
        --camera-distance)
            camera_distance_m="$2"
            shift 2
            ;;
        --camera-height)
            camera_height_m="$2"
            shift 2
            ;;
        --camera-target-height)
            camera_target_height_m="$2"
            shift 2
            ;;
        --camera-smoothing)
            camera_smoothing="$2"
            shift 2
            ;;
        --camera-max-yaw-rate)
            camera_max_yaw_rate_dps="$2"
            shift 2
            ;;
        --demo-robot-marker)
            demo_robot_marker=true
            shift
            ;;
        --demo-robot-marker-radius)
            demo_robot_marker_radius_m="$2"
            shift 2
            ;;
        --real-time-factor)
            real_time_factor="$2"
            shift 2
            ;;
        --command-timeout)
            command_timeout_s="$2"
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

case "${profile}" in
    v1)
        ;;
    v2)
        if [[ "${scene_variant}" == "warehouse" ]]; then
            scene_variant=long_warehouse
        fi
        if [[ "${layout_seed}" == "7" ]]; then
            layout_seed=101
        fi
        ;;
    *)
        echo "Invalid profile: ${profile}. Expected v1 or v2." >&2
        exit 2
        ;;
esac

ros_launch_pid=""
goal_config_file=""
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

    if [[ -n "${goal_config_file}" && -f "${goal_config_file}" ]]; then
        rm -f "${goal_config_file}"
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
goal_config_file="$(mktemp /tmp/aerostrike_goal.XXXXXX.yaml)"
policy_config_file="${workspace_root}/aerostrike_pkg/config/policy_node.yaml"
observation_config_file="${workspace_root}/aerostrike_pkg/config/observation_builder.yaml"
command_config_file="${workspace_root}/aerostrike_pkg/config/command_adapter.yaml"
scene_config_file="${workspace_root}/configs/scene_variants.yaml"
if [[ "${profile}" == "v2" ]]; then
    policy_config_file="${workspace_root}/aerostrike_pkg/config/policy_node_v2.yaml"
    observation_config_file="${workspace_root}/aerostrike_pkg/config/observation_builder_v2.yaml"
    command_config_file="${workspace_root}/aerostrike_pkg/config/command_adapter_v2.yaml"
    scene_config_file="${workspace_root}/configs/scene_variants_v2.yaml"
fi
PYTHONPATH="${workspace_root}/aerostrike_lab:${PYTHONPATH:-}" python3 - "${layout_seed}" "${scene_variant}" "${goal_config_file}" "${scene_config_file}" <<'PY'
import sys
from pathlib import Path

from aerostrike_lab.scenes.warehouse import load_warehouse_scene_settings, sample_warehouse_layout

layout_seed = int(sys.argv[1])
scene_variant = sys.argv[2]
goal_config_path = Path(sys.argv[3])
scene_config_path = Path(sys.argv[4])
layout = sample_warehouse_layout(
    settings=load_warehouse_scene_settings(scene_config_path),
    seed=layout_seed,
    scene_variant=scene_variant,
)
goal = layout.goal_position
goal_config_path.write_text(
    "goal_publisher:\n"
    "  ros__parameters:\n"
    '    goal_topic: "/aerostrike/goal"\n'
    '    frame_id: "world"\n'
    f"    goal_x: {goal[0]}\n"
    f"    goal_y: {goal[1]}\n"
    f"    goal_z: {goal[2]}\n"
    "    publish_rate_hz: 2.0\n",
    encoding="utf-8",
)
print(
    "[demo] Goal config: "
    f"scene_variant={scene_variant} layout_seed={layout_seed} "
    f"goal=({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}) "
    f"file={goal_config_path}"
)
PY
setsid ros2 launch aerostrike_pkg policy_runtime.launch.xml \
    goal_config_file:="${goal_config_file}" \
    policy_config_file:="${policy_config_file}" \
    observation_config_file:="${observation_config_file}" \
    command_config_file:="${command_config_file}" &
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
    --profile "${profile}"
    --layout_seed "${layout_seed}"
    --scene_variant "${scene_variant}"
    --camera_mode "${camera_mode}"
    --camera_distance_m "${camera_distance_m}"
    --camera_height_m "${camera_height_m}"
    --camera_target_height_m "${camera_target_height_m}"
    --camera_smoothing "${camera_smoothing}"
    --camera_max_yaw_rate_dps "${camera_max_yaw_rate_dps}"
    --demo_robot_marker_radius_m "${demo_robot_marker_radius_m}"
    --steps "${steps}"
    --real_time_factor "${real_time_factor}"
    --command_timeout_s "${command_timeout_s}"
)

if [[ "${visible}" == "true" ]]; then
    isaac_args+=(--visible)
fi
if [[ "${demo_robot_marker}" == "true" ]]; then
    isaac_args+=(--demo_robot_marker)
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
