#!/usr/bin/env python3
"""Initialise AeroStrike Obsidian vault with project baseline content."""

import os
import sys
from pathlib import Path
from datetime import date

VAULT = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./akasha"))

files = {}

# ─── 00_INDEX.md ─────────────────────────────────────────────────────────────
files["00_INDEX.md"] = f"""---
tags: [index, project]
updated: {date.today()}
---

# AeroStrike — Project Index

> RL-augmented high-speed drone navigation | Isaac Sim + Isaac Lab + ROS 2

## Status: Week 1 — Base System

## Quick links
- [[Tasks/active|Active tasks]]
- [[Milestones/roadmap|4-week roadmap]]
- [[Environment/stack|Tech stack & versions]]
- [[RL/reward_design|RL reward design]]
- [[Isaac_Sim/warehouse_env|Warehouse environment]]

## Architecture
[Goal Position]
↓
[Waypoint Generator — ROS 2]
↓
[RL Local Policy — Isaac Lab]
↓
[Drone Controller]
↓
[Isaac Sim Physics]

## Success criteria (Month 1)
- [ ] Move from A to B
- [ ] Obstacle avoidance ≥ 80% success rate
- [ ] Speed 3–5 m/s
- [ ] Stable non-random behaviour

## Sessions log (latest first)
```dataview
LIST file.mtime
FROM "Sessions"
SORT file.mtime DESC
LIMIT 5
```

## Open problems
```dataview
LIST
FROM "Problems"
WHERE status = "open"
```

## Active tasks
```dataview
TASK
FROM "Tasks"
WHERE !completed
LIMIT 10
```
"""

# ─── Tasks/active.md ─────────────────────────────────────────────────────────
files["Tasks/active.md"] = f"""---
tags: [tasks]
week: 1
updated: {date.today()}
---

# Active Tasks — Week 1

## In Progress
- [ ] Create warehouse scene in Isaac Sim #isaac-sim #week1
- [ ] Implement basic drone movement (velocity control interface) #core #week1
- [ ] Connect goal point → direction vector logic #week1

## Backlog — Week 1
- [ ] Test raycast sensor output from Isaac Sim #sensors
- [ ] Verify ground truth pose access in Isaac Sim #localization
- [ ] Set up Isaac Lab task skeleton #rl #week2-prep

## Done
<!-- Agent moves completed tasks here with completion date -->
"""

# ─── Tasks/backlog.md ────────────────────────────────────────────────────────
files["Tasks/backlog.md"] = f"""---
tags: [tasks, backlog]
---

# Backlog (Future Weeks)

## Week 2 — RL Initialisation
- [ ] Integrate Isaac Lab into project
- [ ] Define observation space (rays + velocity + goal dir)
- [ ] Define action space (vx, vy, vz)
- [ ] Train obstacle avoidance (no goal yet)
- [ ] Evaluate: drone avoids obstacles

## Week 3 — Goal Integration
- [ ] Add goal direction to observations
- [ ] Tune reward weights
- [ ] Add speed maintenance penalty
- [ ] Evaluate: drone moves toward goal

## Week 4 — Integration
- [ ] Add ROS 2 waypoint publisher
- [ ] Wire RL policy into ROS 2 pipeline
- [ ] Collect metrics (success rate, collision rate, avg speed)
- [ ] Record 60–90s demo video

## Post-MVP Extensions
- [ ] V2: Replace rays with depth images
- [ ] V3: Add SLAM
- [ ] V4: Dynamic obstacles
- [ ] V5: Multi-drone
"""

# ─── Milestones/roadmap.md ───────────────────────────────────────────────────
files["Milestones/roadmap.md"] = f"""---
tags: [milestones, roadmap]
---

# 4-Week Roadmap

| Week | Goal | Key deliverable | Status |
|------|------|----------------|--------|
| 1 | Base system | Drone moves in scene | In progress |
| 2 | RL init | Drone avoids obstacles | Not started |
| 3 | Goal integration | Drone moves toward goal | Not started |
| 4 | Full integration | Complete MVP + demo | Not started |

## Milestone details

### M0: Project setup
- Workspace structure created
- Obsidian vault initialised
- Codex CLI configured with MCP

### M1: Base system (Week 1)
- **Due:** End of Week 1
- Criteria:
  - [ ] Warehouse scene loaded in Isaac Sim
  - [ ] Drone spawns and moves via velocity commands
  - [ ] Goal position accepted, direction computed

### M2: RL obstacle avoidance (Week 2)
- **Due:** End of Week 2
- Criteria:
  - [ ] Isaac Lab task defined
  - [ ] Policy trained: avoidance without goal
  - [ ] Collision rate < 50% in evaluation

### M3: Goal-directed navigation (Week 3)
- **Due:** End of Week 3
- Criteria:
  - [ ] Policy reaches goal in > 60% of runs
  - [ ] Speed maintained at 3–5 m/s
  - [ ] Reward function stable

### M4: MVP complete (Week 4)
- **Due:** End of Week 4
- Criteria:
  - [ ] ROS 2 + RL pipeline integrated
  - [ ] Success rate ≥ 80%
  - [ ] Demo video recorded
  - [ ] Metrics documented
"""

# ─── Environment/stack.md ────────────────────────────────────────────────────
files["Environment/stack.md"] = f"""---
tags: [environment, versions]
updated: {date.today()}
---

# Tech Stack & Versions

## Fill in your actual versions on first setup

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Ubuntu 22.04 | |
| Python | 3.11 | |
| ROS 2 | Humble | |
| Isaac Sim | 5.1.0 | |
| Isaac Lab | 1.x | |
| CUDA | xx.x | check: `nvcc --version` |
| PyTorch | x.x.x | |
| Codex CLI | x.x.x | `codex --version` |
| Node.js | 20.x | |

## Environment variables (names only, not values)
- `OPENAI_API_KEY`
- `OBSIDIAN_API_KEY`
- `OBSIDIAN_PORT`
- `OBSIDIAN_VAULT_PATH`
- `ROS_DOMAIN_ID`

## Isaac Sim setup notes
<!-- Agent fills this in as setup progresses -->

## Known environment issues
<!-- Agent logs here when environment problems occur -->
"""

# ─── RL/reward_design.md ─────────────────────────────────────────────────────
files["RL/reward_design.md"] = f"""---
tags: [rl, reward]
status: draft
---

# Reward Function Design

## Formula
R = + w1 * progress_to_goal
+ w2 * forward_velocity
- w3 * collision_penalty
- w4 * proximity_penalty
- w5 * instability_penalty

## Initial weights (to be tuned)

| Weight | Value | Rationale |
|--------|-------|-----------|
| w1 (progress) | 1.0 | primary objective |
| w2 (velocity) | 0.3 | prevent speed collapse |
| w3 (collision) | 10.0 | hard penalty |
| w4 (proximity) | 0.5 | soft penalty, scales with closeness |
| w5 (instability) | 0.2 | jerk/oscillation penalty |

## Known risks
- **Speed collapse**: RL tends to slow down → increase w2 if observed
- **Jitter**: oscillating actions → increase w5
- **Goal-ignoring**: drone circles → increase w1

## Tuning history
<!-- Agent logs experiments here -->
| Date | Change | Result |
|------|--------|--------|
| {date.today()} | Initial design | baseline |

## Observation space
- `rays[0:N]` — N distance readings (16–32 rays, LiDAR-like raycast)
- `velocity[3]` — (vx, vy, vz)
- `goal_dir[3]` — normalised direction to goal (dx, dy, dz)

## Action space
- `action[3]` — desired velocity (vx, vy, vz)
"""

# ─── Isaac_Sim/warehouse_env.md ───────────────────────────────────────────────
files["Isaac_Sim/warehouse_env.md"] = f"""---
tags: [isaac-sim, environment]
status: planned
---

# Warehouse Environment

## Choice rationale
Option A (Industrial warehouse) selected over forest:
- Structured geometry → easier RL convergence
- Consistent obstacle spacing
- Clear visual feedback for debugging

## Planned layout
- Walls defining arena boundary
- Vertical beams / pillars (static obstacles)
- Box stacks (static obstacles)
- Open floor (drone operates at ~1–2m height)

## Scene file
- Path: `isaac_lab/envs/warehouse/`
- USD file: `<!-- to be created -->`

## Spawn configuration
- Drone spawn: random within safe zone
- Goal position: random, min distance 10m from spawn
- Obstacle density: medium (tune for RL convergence)

## Sensor setup
- Raycast count: 16 initially (expand to 32 if needed)
- Ray pattern: horizontal fan (±90° in XY plane) + 2 downward
- Max range: 10m

## Change log
<!-- Agent logs every change to scene here -->
| Date | Change | Session |
|------|--------|---------|
"""

# ─── Problems/ (empty placeholder) ──────────────────────────────────────────
files["Problems/README.md"] = """---
tags: [problems]
---

# Problems & Solutions

Each problem gets its own note: `Problems/short-title.md`

## Format
status: open | resolved | wontfix
severity: blocker | major | minor

## Index
```dataview
TABLE status, severity, file.mtime as "Updated"
FROM "Problems"
WHERE file.name != "README"
SORT file.mtime DESC
```
"""

# ─── Ideas/README.md ─────────────────────────────────────────────────────────
files["Ideas/README.md"] = """---
tags: [ideas]
---

# Ideas & Proposals

## Post-MVP extensions (from spec)
- [[Ideas/v2-depth-images|V2: depth image observations]]
- [[Ideas/v3-slam|V3: SLAM integration]]
- [[Ideas/v4-dynamic-obstacles|V4: dynamic obstacles]]
- [[Ideas/v5-multi-drone|V5: multi-drone coordination]]

## Open ideas
```dataview
LIST
FROM "Ideas"
WHERE file.name != "README"
SORT file.mtime DESC
```
"""

# ─── Write all files ─────────────────────────────────────────────────────────
for rel_path, content in files.items():
    full_path = VAULT / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    print(f"  ✓  {rel_path}")

print(f"\nVault initialised at: {VAULT.resolve()}")
print("Open the 'akasha/' folder as a vault in Obsidian.")