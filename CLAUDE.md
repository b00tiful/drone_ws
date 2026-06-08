# AeroStrike Agent — Rules & Context

## Project
AeroStrike: RL-augmented autonomous drone navigation.
Stack: Isaac Sim + Isaac Lab + ROS 2 + Python RL.
Goal: Working MVP in 4 weeks (obstacle avoidance + goal navigation at 3–5 m/s).

## Workspace layout

```
drone_ws/
├── README.md
├── setup_env.sh                    # Environment setup script
│
├── aerostrike_lab/                  # Isaac Lab extension (RL training)
│   ├── pyproject.toml
│   └── aerostrike_lab/
│       ├── __init__.py
│       ├── tasks/
│       │   ├── __init__.py
│       │   └── navigation/
│       │       ├── __init__.py          # gym.register()
│       │       ├── nav_env.py           # DirectRLEnv — THE core file
│       │       └── nav_env_cfg.py       # DirectRLEnvCfg
│       ├── assets/
│       │   ├── __init__.py
│       │   └── quadrotor.py             # Drone asset config (Crazyflie USD)
│       └── scenes/
│           ├── __init__.py
│           └── warehouse.py             # Procedural warehouse scene generator
│
├── scripts/
│   ├── train.py                     # Launch RL training
│   ├── play.py                      # Evaluate trained policy
│   ├── benchmark.py                 # Collect metrics (success rate, speed, etc.)
│   └── export_policy.py             # Export ONNX for ROS 2 inference
│
├── configs/
│   ├── ppo_aerostrike.yaml          # PPO hyperparameters (skrl)
│   └── scene_variants.yaml          # Obstacle layout configs
│
├── aerostrike_pkg/                   # ROS 2 package (Week 4)
│   ├── package.xml
│   ├── setup.py
│   ├── aerostrike_pkg/
│   │   ├── __init__.py
│   │   ├── goal_publisher.py         # Publishes goal position
│   │   ├── waypoint_generator.py     # Simple direction vector planner
│   │   ├── policy_node.py            # Runs trained ONNX policy
│   │   └── metrics_logger.py         # Logs success/collision/speed
│   └── launch/
│       └── aerostrike_launch.py
│
├── logs/                             # TensorBoard logs
├── akasha                            # Obsidian Vault. Your ultimate knowledge base
└── checkpoints/                      # Saved model weights
```

## MANDATORY: Documentation rules

You MUST follow these rules on every non-trivial action:

### After writing or modifying code:
1. Call `obsidian_append_note` to log the action to `Sessions/YYYY-MM-DD.md`
2. If a new ROS 2 node was created → create `ROS2_Nodes/<node_name>.md`
3. If Isaac Sim scene/environment changed → update `Isaac_Sim/<env_name>.md`
4. If a bug was encountered → create or update `Problems/<short_title>.md`
5. Update task status in `Tasks/active.md`

### Before starting any task:
1. Read `akasha/00_INDEX.md` — full project state
2. Read `Tasks/active.md` — current tasks
3. Read last 2 entries in `Sessions/` — recent progress
4. Read relevant domain file (e.g., `RL/reward_tuning.md` if working on rewards)

### Never skip documentation. The vault IS your memory.

## Current project phase
→ Week 1: Base system (scene + drone movement + goal connection)

## Core Development Principles (Karpathy Style)

### 1. Think Before Coding
- **Never make assumptions.** Do not guess implementation details regarding system architecture, hardware constraints, or communication protocols.
- **Evaluate trade-offs.** Before writing any code, explicitly outline the pros and cons of the proposed solution.
- **Stop and ask.** If a requirement, sensor metric, or control loop behavior is ambiguous or poorly defined, STOP immediately and ask for clarification.

### 2. Simplicity First
- **Minimalist implementation.** Write the absolute minimum amount of code required to achieve the goal. 
- **No over-engineering.** Avoid premature abstractions, complex design patterns, or building features "for the future." Keep code flat and straightforward.
- **Readability over cleverness.** Prioritize clean, self-documenting code over complex, condensed syntax.

### 3. Surgical Changes
- **Strict scoping.** Modify only the files and lines of code that are directly relevant to the current task.
- **No unauthorized refactoring.** Do not touch, "clean up," or reformat adjacent code blocks or unrelated files unless explicitly requested.
- **Compact diffs.** Keep changes localized and minimal to ensure effortless debugging and clean git commits.

### 4. Goal-Driven Execution
- **Define success first.** Identify clear verification criteria (e.g., node stability, loop rate compliance, or precise state transitions) before execution.
- **Focus on the core objective.** Deliver functional correctness that matches the exact goal specified, ensuring the system remains robust and predictable.

## Key technical constraints
- Use Isaac Lab (NOT Isaac Gym)
- Observation space: rays (16–32) + velocity + goal direction. NO images.
- Action space: desired velocity (vx, vy, vz)
- Reward: progress_to_goal + forward_velocity - collision - proximity - instability
- Environment: Industrial warehouse (Option A)
- Localization: ground truth pose only

## Coding standards
- ROS 2 Humble
- Python 3.11
- Type hints everywhere
- Every node must have a corresponding `akasha/ROS2_Nodes/` entry
- All config values in YAML, never hardcoded

## Git workflow

- Never run `git commit` on your own.
- After completing a task, show me the diff (`git diff --staged` or `git status`).
- Suggest a commit message, but wait for my confirmation or edits before proceeding.
- Only run `git push` after I explicitly say so.
- Use conventional commits: feat:, fix:, refactor:, etc.

## Session log format
When logging to Sessions/, use this format:

HH:MM — <action title>
Task: <what was the goal>
Done: <what was actually done>
Files changed: path/to/file.py, path/to/other.py
Result: <outcome — success / partial / failed>
Problems: <any issues encountered>
Next: <immediate next step>

## On failure / blockers
If something doesn't work after 2 attempts:
1. Log the problem to `Problems/<title>.md` with full details
2. Note it in the current session log
3. Ask for clarification before continuing

## Linking & graph rules

Every note MUST contain wiki-links to related notes. This builds the knowledge graph.

**Always link when:**
- Creating a session → link to tasks worked on, problems encountered, nodes created
  `[[Tasks/active]]`, `[[ROS2_Nodes/drone_goal_publisher]]`, `[[Problems/cuda-oom]]`
- Creating a ROS node → link to related Isaac Sim env, RL task, session that created it
  `[[Isaac_Sim/warehouse_env]]`, `[[RL/nav_task]]`, `[[Sessions/2025-01-15]]`
- Logging a problem → link to session, node or file where it appeared
  `[[Sessions/2025-01-15]]`, `[[ROS2_Nodes/policy_node]]`
- Completing a milestone → link to all sessions and notes that contributed to it
- Creating an idea → link to the problem or observation that triggered it
  `[[Problems/speed-collapse]]` → `[[Ideas/adaptive-reward-scaling]]`

**Minimum links per note type:**
| Note type | Min links |
|-----------|-----------|
| Session | 2 (tasks + nodes/files touched) |
| ROS node | 2 (isaac env + session created) |
| Isaac Sim env | 1 (related RL task) |
| Isaac Lab task | 2 (env + reward design) |
| Problem | 2 (session + component) |
| Idea | 1 (trigger: problem or observation) |
| Milestone | 3+ (sessions + tasks + results) |

**Format:** always use `[[path/to/note|display name]]` with display name for readability.
Example: `[[Sessions/2025-01-15|Session: warehouse scene setup]]`

<!-- CODEGRAPH_START -->
## CodeGraph

This project has a CodeGraph MCP server (`codegraph_*` tools) configured. CodeGraph is a tree-sitter-parsed knowledge graph of every symbol, edge, and file. Reads are sub-millisecond and return structural information grep cannot.

### When to prefer codegraph over native search

Use codegraph for **structural** questions — what calls what, what would break, where is X defined, what is X's signature. Use native grep/read only for **literal text** queries (string contents, comments, log messages) or after you already have a specific file open.

| Question | Tool |
|---|---|
| "Where is X defined?" / "Find symbol named X" | `codegraph_search` |
| "What calls function Y?" | `codegraph_callers` |
| "What does Y call?" | `codegraph_callees` |
| "What would break if I changed Z?" | `codegraph_impact` |
| "Show me Y's signature / source / docstring" | `codegraph_node` |
| "Give me focused context for a task/area" | `codegraph_context` |
| "See several related symbols' source at once" | `codegraph_explore` |
| "What files exist under path/" | `codegraph_files` |
| "Is the index healthy?" | `codegraph_status` |

### Rules of thumb

- **Answer directly — don't delegate exploration.** For "how does X work" / architecture / trace questions, answer with 2-3 codegraph calls: `codegraph_context` first, then ONE `codegraph_explore` for the source of the symbols it surfaces. Codegraph IS the pre-built index, so spawning a separate file-reading sub-task/agent — or running a grep + read loop — repeats work codegraph already did and costs more for the same answer.
- **Trust codegraph results.** They come from a full AST parse. Do NOT re-verify them with grep — that's slower, less accurate, and wastes context.
- **Don't grep first** when looking up a symbol by name. `codegraph_search` is faster and returns kind + location + signature in one call.
- **Don't chain `codegraph_search` + `codegraph_node`** when you just want context — `codegraph_context` is one call.
- **Don't loop `codegraph_node` over many symbols** — one `codegraph_explore` call returns several symbols' source grouped in a single capped call, while each separate node/Read call re-reads the whole context and costs far more.
- **Index lag**: the file watcher debounces ~500ms behind writes; don't re-query immediately after editing a file in the same turn.

### If `.codegraph/` doesn't exist

The MCP server returns "not initialized." Ask the user: *"I notice this project doesn't have CodeGraph initialized. Want me to run `codegraph init -i` to build the index?"*
<!-- CODEGRAPH_END -->
