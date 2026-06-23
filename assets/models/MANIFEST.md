# AeroStrike Model Assets

Related: [[akasha/Isaac_Sim/warehouse_env|warehouse environment]] [[akasha/Tasks/active|active tasks]]

## Kenney Factory Kit

- Source: https://kenney.nl/assets/factory-kit
- Download URL used: https://kenney.nl/media/pages/assets/factory-kit/edaac9d4f6-1777639602/kenney_factory-kit_3.0.zip
- Version: 3.0
- License: Creative Commons CC0, copied in `kenney_factory_kit/License.txt`
- Imported date: 2026-06-23
- Purpose: optimized real 3D industrial props for Demo V1 cinematic hallway visuals.

## Selected Assets

The repo keeps a curated subset instead of the full pack. Each selected FBX is small enough for repeated visual-only instancing in Isaac Sim; the GLB copy is kept for preview/conversion fallback.

- `box-large`, `box-long`, `box-small`, `box-wide`
- `catwalk-straight`
- `conveyor-long-stripe-sides`, `conveyor-bars-stripe-fence`, `conveyor-corner`
- `crane`, `crane-lift`
- `hopper-high-square`
- `machine-bed`, `machine-fortified`, `machine-window`
- `pipe-large-long`, `pipe-large-valve`
- `robot-arm-a`
- `scanner-high`
- `screen-panel-wide`
- `structure-window-wide`, `structure-yellow-high`
- `warning-traffic`

## Runtime Use

- Source meshes live under `assets/models/kenney_factory_kit/fbx`.
- Isaac Lab converts selected FBX meshes to USD lazily at runtime under ignored `assets/models/kenney_factory_kit/usd_cache`.
- Scene code spawns these as presentation-only models under `/World/Visuals`; they are not added to the policy RayCaster target list.
