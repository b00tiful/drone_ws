# AeroStrike

AeroStrike is an RL-augmented autonomous drone navigation MVP built on Isaac Sim,
Isaac Lab, ROS 2, and Python. The Week 1 target is a base system where a drone
spawns in an industrial warehouse scene, moves through a velocity interface, and
connects goal positions to direction vectors.

## Current Phase

- Week 3: goal integration
- Primary stack: Isaac Sim 5.1.0, Isaac Lab 1.x, ROS 2 Humble, Python 3.11
- Navigation MVP target: obstacle avoidance plus goal navigation at 3-5 m/s+

## Workspace

```text
drone_ws/
├── aerostrike_lab/       # Isaac Lab extension and RL environment
├── aerostrike_pkg/       # ROS 2 integration package, added in Week 4
├── configs/              # Training and scene configuration
├── scripts/              # Training, evaluation, benchmark, export entrypoints
├── logs/                 # TensorBoard and runtime logs
├── checkpoints/          # Saved policy weights
└── akasha/               # Obsidian project memory
```

## Setup

Source `setup_env.sh` from the workspace root after installing Isaac Sim,
Isaac Lab, ROS 2 Humble, and Python dependencies:

```bash
source setup_env.sh
```

Then install the Isaac Lab extension in editable mode:

```bash
python3 -m pip install -e aerostrike_lab
```

## Development Notes

- Use Isaac Lab DirectRLEnv for the navigation task.
- Observation space uses ray distances, velocity, and goal direction. Images are
  intentionally out of scope for the MVP.
- Action space is desired velocity `(vx, vy, vz)`.
- Project progress and implementation decisions are tracked in `akasha/`.
