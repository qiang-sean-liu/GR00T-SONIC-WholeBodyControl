# SONIC Data Collection Guide

This guide covers everything needed to collect teleoperation data for SONIC model training — from camera setup to episode management, output formats, and dataset structure.

```{admonition} Prerequisites
:class: note
Complete the [Quick Start](../getting_started/quickstart), [PICO Setup](../getting_started/vr_teleop_setup), and review the [Whole-body Teleoperation Guide](teleoperation.md) before collecting data.
```

---

## Overview

The data collection pipeline captures synchronized robot state, end-effector poses, actions, teleoperation commands, and camera video during whole-body teleoperation. Data is saved in **LeRobot dataset format** (Parquet + MP4) and is directly compatible with the SONIC training pipeline.

Two collection modes are available:

| Mode | Script | Use case |
|------|--------|----------|
| **Simulation** | `run_sync_sim_data_collection.py` | Development, debugging, sim-to-real data |
| **Real robot** | `run_g1_data_exporter.py` | Production data collection on physical G1 |

---

## Cameras

The pipeline supports four physical camera positions via `ComposedCameraSensor`. Each position is independently optional and configurable.

| Dataset key | Mount position | Hardware options | Default |
|-------------|---------------|-----------------|---------|
| `observation.images.ego_view` | Chest / torso-mounted | `oak`, `realsense`, `zed` | `oak` (always on) |
| `observation.images.head` | Head-mounted (D435 in URDF) | `oak`, `oak_mono`, `realsense`, `zed` | None |
| `observation.images.left_wrist` | Left wrist | `oak`, `realsense`, `zed` | None |
| `observation.images.right_wrist` | Right wrist | `oak`, `realsense`, `zed` | None |

An optional **stereo pair** adds two additional streams (flag `--add_stereo_camera`):
- `observation.images.ego_view_left_mono`
- `observation.images.ego_view_right_mono`

**Default configuration** (`DataExporterConfig`): only `ego_view` via OAK camera is recorded. `add_stereo_camera` defaults to `True` in the real-robot config, adding the stereo pair. Head and wrist cameras must be explicitly enabled.

All camera images are captured at **640 × 480 RGB**, 20–30 fps depending on mode.

### Starting the Camera Server

The camera server runs as a separate process and is accessed by the data exporter as a client over ZMQ:

```bash
# Start composed camera server (adjust camera types as needed)
python -m decoupled_wbc.control.sensor.composed_camera \
    --ego_view_camera realsense \
    --head_camera realsense \
    --port 5555
```

---

## Recorded Data Fields

Each frame in the dataset contains:

| Key | Type | Shape | Content |
|-----|------|-------|---------|
| `observation.state` | float64 | `(43,)` | All joint angles |
| `observation.eef_state` | float64 | `(14,)` | Left+right wrist pose (pos xyz + quat xyzw) |
| `observation.images.*` | uint8 video | `(480, 640, 3)` | RGB camera frames (one entry per enabled camera) |
| `observation.img_state_delta` | float32 | `(1,)` | Timing delta between image and proprioception (ms) |
| `action` | float64 | `(43,)` | Joint position commands |
| `action.eef` | float64 | `(14,)` | End-effector target pose (left+right) |
| `teleop.navigate_command` | float64 | `(3,)` | Base velocity `[lin_x, lin_y, ang_z]` |
| `teleop.base_height_command` | float64 | `(1,)` | Base height adjustment |

The `modality_config` maps state/action indices to named joint groups:
`left_leg`, `right_leg`, `waist`, `left_arm`, `left_hand`, `right_arm`, `right_hand`, plus wrist position and quaternion sub-fields.

---

## Output Format and Directory Structure

Data is stored in **LeRobot dataset format**:

```
<root_output_dir>/<dataset_name>/
├── meta/
│   ├── info.json          # episode count, task list, script config, data collection metadata
│   ├── modality.json      # joint group → index mapping for state/action
│   └── features.json      # feature schema (dtype, shape, names)
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── ego_view/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── ego_view_left_mono/   # if stereo enabled
        └── ego_view_right_mono/  # if stereo enabled
```

- **Parquet files**: one per episode, contain all non-video fields plus frame indices and timestamps
- **MP4 files**: h264-encoded, one per camera per episode
- Episodes flagged as discarded are tracked in `meta/info.json` and excluded from training

---

## Real Robot Data Collection

### Setup

Requires the full SONIC stack to be running. Start each component in a separate terminal:

**Terminal 1 — C++ Deployment**:
```bash
cd gear_sonic_deploy
bash deploy.sh real --input-type zmq_manager
```

**Terminal 2 — PICO Teleop Streamer**:
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

**Terminal 3 — Camera Server**:
```bash
source .venv_teleop/bin/activate
python -m decoupled_wbc.control.sensor.composed_camera \
    --ego_view_camera oak \
    --port 5555
```

**Terminal 4 — Data Exporter**:
```bash
source .venv_teleop/bin/activate
python decoupled_wbc/control/main/teleop/run_g1_data_exporter.py \
    --dataset_name my_dataset \
    --task_prompt "pick up the apple and place it on the plate" \
    --robot_id G1_001 \
    --teleoperator_username alice \
    --root_output_dir outputs
```

### Key Configuration Options (`DataExporterConfig`)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_name` | auto-generated | Dataset folder name; appends to existing if present |
| `--task_prompt` | `"demo"` | Natural language task description saved with each episode |
| `--root_output_dir` | `"outputs"` | Parent directory for all datasets |
| `--robot_id` | None | Robot serial/ID tag stored in metadata |
| `--teleoperator_username` | None | Operator name stored in metadata |
| `--add_stereo_camera` | `True` | Include stereo camera pair |
| `--camera_host` | `"localhost"` | Camera server host |
| `--camera_port` | `5555` | Camera server ZMQ port |
| `--fps` | `20.0` | Data collection rate (Hz) |
| `--img_stream_viewer` | `False` | Open live matplotlib window to preview camera |
| `--text_to_speech` | `True` | Voice feedback during recording |

If `--dataset_name` is not provided, the script will prompt for it interactively, or auto-generate a timestamped name.

### Keyboard Controls (during collection)

| Key | Action |
|-----|--------|
| `c` | **Toggle recording** — start a new episode, or stop and queue for saving |
| `x` | **Discard** current episode (marks as failed, excluded from training) |
| `Ctrl+C` | Exit — auto-saves or discards the in-progress episode |

A timing monitor warns if the image–state synchronization delta is too high; discard (`x`) those episodes.

---

## Simulation Data Collection

### Setup

**Terminal 1 — MuJoCo Simulator**:
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**Terminal 2 — C++ Deployment**:
```bash
cd gear_sonic_deploy
bash deploy.sh sim --input-type zmq_manager
```

**Terminal 3 — PICO Teleop Streamer**:
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

**Terminal 4 — Sim Data Collection**:
```bash
source .venv_teleop/bin/activate
python decoupled_wbc/control/main/teleop/run_sync_sim_data_collection.py \
    --task_name PnPBottle
```

### Key Configuration Options (`SyncSimDataCollectionConfig`)

| Flag | Default | Description |
|------|---------|-------------|
| `--task_name` | `"GroundOnly"` | Task environment (`GroundOnly`, `PnPBottle`, etc.) |
| `--manual_control` | `False` | If True, recording must be triggered manually instead of auto-saving on task completion |
| `--save_img_obs` | `False` | Save image observations (sim rendering) |
| `--success_hold_steps` | `50` | Steps to record after task success before saving |
| `--enable_onscreen` | `True` | Show MuJoCo viewer window |
| `--renderer` | `"mjviewer"` | Renderer: `mjviewer`, `mujoco`, or `rerun` |

In simulation, episode saving is **automatic** by default (triggered by task completion detection). Set `--manual_control True` to control episodes manually with the keyboard.

---

## Episode Lifecycle

```
IDLE → (trigger) → RECORDING → (trigger) → NEED_TO_SAVE → IDLE
                                    ↓
                              (discard) → IDLE
```

- **IDLE**: waiting to record
- **RECORDING**: actively collecting frames
- **NEED_TO_SAVE**: episode complete, being encoded and written to disk
- Episodes saved to disk are appended to the dataset; discarded episodes are logged but not used in training

---

## XRoboToolkit Recording (Pose-only)

The XRoboToolkit PC app (`RobotDataRecorder`) provides a separate recording facility that captures **tracking/pose data only** — no camera images. It is suitable for logging raw PICO motion capture independently of the SONIC pipeline.

**Output directory**: timestamped folder `yyyy-MM-dd_HH-mm-ss/` created in the working directory of `RoboticsServiceProcess` (typically `/home/horizon/pico/XRoboToolkit/`).

| File | Format | Content |
|------|--------|---------|
| `head.txt` | JSON lines | Head pose per frame |
| `hand.txt` | JSON lines | Hand tracking (26 joints × 2 hands) |
| `controller.txt` | JSON lines | Trigger, grip, buttons, joystick axes |
| `body.csv` | CSV | 24 body joints × (x,y,z,qx,qy,qz,qw) with `localTimeStamp` and `remoteTimeStamp` |
| `motion.txt` | JSON lines | Motion tracker data |

The `localTimeStamp` column in `body.csv` can be used to align XRoboToolkit recordings with external camera recordings (e.g. D435 via ROS bag or `run_webcam_recorder.py`).

**Note**: XRoboToolkit recording does **not** capture camera images. For synchronized camera + pose recording for model training, use the SONIC pipeline (`run_g1_data_exporter.py`) instead.

---

## Tips for Quality Data

1. **Wear tight-fitting clothing** — Required for reliable foot tracker visibility (see [Teleoperation Guide](teleoperation.md#clothing-requirements))
2. **Recalibrate often** — PICO tracking drifts over time; recalibrate between episodes
3. **Discard jittery episodes** — Use `x` immediately if you see tracking glitches or large `img_state_delta` warnings
4. **Keep WiFi latency < 10ms** — High latency causes unnatural robot motion and poor training data
5. **Consistent task framing** — Position objects in the same region across episodes for better policy generalization
6. **Short, clean episodes** — Complete the task cleanly; avoid lingering or correcting mid-episode

---

## Related Documentation

- [Whole-body Teleoperation Guide](teleoperation.md) — setup, safety, movement practices
- [Training Data](training_data.md) — BONES-SEED dataset and training pipeline
- [VR Teleop Setup](../getting_started/vr_teleop_setup.md) — PICO hardware setup
- [Troubleshooting](troubleshooting.md) — common issues and fixes
