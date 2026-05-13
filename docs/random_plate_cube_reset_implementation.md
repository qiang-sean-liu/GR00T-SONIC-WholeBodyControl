# Random Plate/Cube Reset Implementation

This document summarizes the branch feature for randomized plate and cube placements during MuJoCo teleoperation data collection.

## Goal

Every time the operator presses Backspace in the MuJoCo viewer:

- Reset the scene.
- Randomize only `plate_body` and `cube_body` on the table.
- Keep the robot, table, cameras, and the rest of the scene unchanged.
- Keep the cube separated from the plate so it does not start stacked on top of the plate.

## Script

Use:

```bash
cd /home/horizon/wrk/SONIC/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop_random_plate_cube.py --enable-image-publish --enable-offscreen --camera-port 5555
```

The script uses the default scene:

```text
gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml
```

That scene already contains:

- `table_body`
- `cube_body`
- `cube_joint`
- `plate_body`

## Placement Logic

The table geometry is derived from the default XML:

- Table center: `(1.5, 0, 0.4)`
- Table top local position: `(0, 0, 0.3)`
- Table top size: `(0.5, 1.0, 0.05)`
- Table top surface height: `z = 0.75`

The randomizer samples:

- Plate `x/y` inside the valid table bounds, with margin for plate radius.
- Cube `x/y` inside the valid table bounds, with margin for cube half-size.
- Plate `z = 0.762`, matching the original scene clearance above the table.
- Cube `z = 0.787`, slightly above the table to avoid degenerate initial contacts.

The cube/plate center distance must be at least:

```text
plate_radius + cube_half_diagonal + extra_clearance
```

This prevents the cube from spawning on top of the plate.

## Reset Handling

The first implementation directly reset and randomized from the MuJoCo viewer key callback. That caused intermittent crashes because the callback can run while the main simulation thread is stepping or rendering.

The current implementation is thread-safe:

- The viewer Backspace callback only sets `_random_plate_cube_reset_requested = True`.
- The actual `mj_resetData()` and object randomization run inside the main sim loop before the next `mj_step`.
- Reset/randomization uses `viewer.lock()` when a passive viewer exists.

## MuJoCo State Updates

On reset, the script:

- Updates `mj_model.body_pos[plate_body]`.
- Updates the cube free-joint position in `mj_data.qpos`.
- Updates the cube default position in `mj_model.qpos0`.
- Resets cube quaternion to identity.
- Clears cube free-joint velocity.
- Clears `qacc_warmstart`.
- Calls `mj_forward()`.

The log prints actual MuJoCo body positions after `mj_forward()`:

```text
[RandomPlateCube] reset N: plate=(...) cube=(...)
```

After every randomized reset, the simulator also writes the latest randomized source XML to:

```text
outputs/latest_random_plate_cube_scene.xml
```

This XML has updated `plate_body` and `cube_body` `pos` values, so it captures the scene state that should be recorded for the next episode.

## Scene XML Recording

For recording datasets with an original-exporter style script plus scene XML artifacts, use:

```bash
cd /home/horizon/wrk/SONIC/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter_original_record_scene_xml.py --task-prompt "pick up the cup"
```

This script is copied from `run_data_exporter_original.py` and adds XML recording only. It copies the configured source scene XML once per saved or discarded episode to:

```text
<dataset_root>/scene_xml/episode_000000.xml
<dataset_root>/scene_xml/episode_000001.xml
...
```

By default it records:

```text
outputs/latest_random_plate_cube_scene.xml
```

The XML is copied when the data-collection toggle starts recording an episode, i.e. the same PICO-side event that maps to `toggle_data_collection` / `c` in the exporter. The save path is:

```text
<dataset_root>/scene_xml/episode_000000.xml
```

Use `--scene-xml-path` to record a different XML file. If the runtime XML does not exist yet, start `run_sim_loop_random_plate_cube.py` first and press Backspace once, or let its startup randomization create the file.
