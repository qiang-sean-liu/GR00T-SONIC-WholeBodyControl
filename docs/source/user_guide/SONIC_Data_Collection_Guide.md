# SONIC Data Collection Guide

This guide covers everything needed to collect teleoperation data for SONIC model training using the gear_sonic pipeline — setup, recording controls, output format, and episode visualization.

```{admonition} Prerequisites
:class: note
Complete the [Quick Start](../getting_started/quickstart), [PICO Setup](../getting_started/vr_teleop_setup), and review the [Whole-body Teleoperation Guide](teleoperation.md) before collecting data.
```

---

## Overview

`record_sonic_teleop.py` captures synchronized data from the gear_sonic pipeline during PICO-driven teleoperation. It subscribes to three ZMQ streams and saves them per episode:

| Stream | Port | Topic | Content |
|--------|------|-------|---------|
| PICO pose | 5556 | `pose` | Human motion (SMPL pose, VR 3-point), controller inputs, hand joints |
| SONIC output | 5557 | `g1_debug` | WBC target/measured joint positions, VR data (heading-corrected) |
| Sim cameras | 5555 | — | JPEG frames from MuJoCo head cameras |
| Base state | 5558 | `base_state` | Ground-truth base position/velocity from MuJoCo physics |

Recording is controlled directly from the PICO headset — no keyboard interaction needed.

For implementation details on receiving PICO data via ZMQ and streaming MuJoCo images back to the headset, see [SONIC ZMQ And PICO Image Streaming Implementation](SONIC_ZMQ_PICO_Image_Streaming_Implementation.md).

---

## Lower Body Control Modes

SONIC supports two lower body control modes, toggled at runtime from the PICO controller — no restart needed.

### POSE mode (default — SMPL / foot tracker driven)

In POSE mode the robot's lower body is driven by **SMPL body fitting** of the operator's full tracked pose (feet, knees, waist via foot trackers). The pipeline:

1. Foot trackers + PICO body tracking → SMPL fit: 24 joint positions in absolute world space
2. SMPL data → pose message (protocol v3) → ZMQ to deploy
3. Deploy-side encoder TRT model builds `smpl_joints_lower_10frame_step1` (9 lower joints × 10 frames = 270 observations)
4. SONIC policy generates leg joint torques from those observations
5. In MuJoCo: torques → ground contact forces → base translates

The result is that **the G1 robot mirrors your physical motion directly** — walking, squatting, kneeling, even walking across the room — without any joystick navigation command. The foot trackers are the key input; no explicit `navigate_cmd` is required for locomotion to occur.

`navigate_cmd_pelvis` records the pelvis velocity (derived from SMPL joint 0 absolute world position) as a proxy for locomotion intent in each frame. When foot trackers are not worn, `navigate_cmd_pelvis` is set to `NaN [3]` and `navigate_cmd` falls back to the joystick source.

### PLANNER_VR_3PT mode (kinematic planner driven)

In PLANNER_VR_3PT mode only **upper body tracking** (L-wrist, R-wrist, Neck) is sent via VR 3-point positions/orientations. A separate kinematic planner network (10 Hz) generates lower body joint trajectories from:

- **`movement_direction`** — left joystick X/Y
- **`speed`** — left joystick magnitude
- **`mode`** — from controller state

In this mode, locomotion requires **explicit joystick input** — walking without moving the joystick produces no lower body motion. `navigate_cmd_joystick` is the relevant field for training.

### Controller mode-switching (no restart needed)

| Buttons | Action |
|---------|--------|
| `A+B+X+Y` | Start policy (OFF → PLANNER) / stop (any mode → OFF) |
| `A+X` | Toggle POSE ↔ PLANNER |
| `B+Y` | Toggle POSE ↔ PLANNER_FROZEN_UPPER_BODY |
| `left_axis_click` | Toggle PLANNER ↔ PLANNER_VR_3PT (within the active planner chain) |

For data collection, **POSE mode is recommended** when the task benefits from natural whole-body motion and the operator wears foot trackers. Use PLANNER_VR_3PT when precise joystick-driven navigation to a location is needed.

---

## Setup (5 terminals)

Follow the [Teleoperation Guide](teleoperation.md#running-mujoco-teleop) to bring up Terminals 1–4,
then add Terminal 5 for the recorder.

**Terminal 1 — MuJoCo Simulator** (with image publishing and base state stream enabled):
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
    --env_name pnp_cube \
    --head_cam \
    --enable_image_publish \
    --enable_offscreen
```

`--env_name` selects the MuJoCo scene. `--head_cam` renders stereo head cameras;
`--enable_image_publish` publishes them over ZMQ on port 5555. Without these flags, no camera
images are saved. `--base_state_port` is optional (default 5558) — it streams ground-truth base
position/velocity from MuJoCo; it is enabled automatically when the recorder requests it.

```{admonition} Match --env_name to the recorder
:class: warning
Pass the **same** `--env_name` value to both `run_sim_loop.py` (Terminal 1) and
`record_sonic_teleop.py` (Terminal 5). The recorder saves it to `meta.json` so the
correct MuJoCo scene is automatically used during conversion and playback. A mismatch
causes the wrong scene to be loaded for episode replay.
```

**Terminal 2 — C++ WBC + SONIC**:
```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim --input-type zmq_manager --enable-model-io-recording
```

**Terminal 3 — PICO Manager**:

POSE mode (default — foot-tracker SMPL drives lower body; pelvis velocity used as `navigate_cmd`):
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --waist_tracking --vis_vr3pt --vis_smpl
```

To force joystick as the navigate_cmd source instead of pelvis (e.g. when not wearing foot trackers):
```bash
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --waist_tracking --vis_vr3pt \
    --navigate_cmd_source joystick
```

The `--navigate_cmd_source` option controls the **active** `navigate_cmd` field only.
`navigate_cmd_joystick` and `navigate_cmd_pelvis` are **always** recorded regardless of this setting;
`navigate_cmd_pelvis` contains `NaN` when foot trackers are not available.

To switch to PLANNER_VR_3PT mode at runtime (no restart needed), use the PICO controller buttons
described in [Lower Body Control Modes](#lower-body-control-modes) above. The same Terminal 3
command is used for all modes — mode switching happens live via controller input.

**Terminal 4 — Headset Video Stream** (optional — stream head cam to PICO):
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/stream_cam_xr.py
```

**Terminal 5 — Recorder**:
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/record_sonic_teleop.py \
    --output_dir ./recordings \
    --env_name pnp_cube \
    --task "Pick up cube and place it in the bin"
```

`--env_name` and `--task` are saved to each episode's `meta.json` and are used automatically
by `convert_sonic_to_lerobot.py` and `playback_lerobot.py` — no need to re-specify them at
conversion time.

To skip camera images (faster, smaller files):
```bash
python gear_sonic/scripts/record_sonic_teleop.py \
    --output_dir ./recordings \
    --env_name pnp_cube \
    --task "Pick up cube and place it in the bin" \
    --no_images
```

**After recording — convert to LeRobot training format:**

`convert_sonic_to_lerobot.py` reads the raw NPZ + JPEG episodes saved by the recorder and
writes a LeRobot dataset (HuggingFace Parquet + H.264 MP4) ready for GR00T N1.5/N1.6
training. It downsamples the ~50 Hz PICO stream to 20 Hz, assembles the 43-DOF state/action
vectors, encodes camera frames as video, and writes the modality config and episode metadata.

If `--env_name` and `--task` were passed to the recorder, `--task` can be omitted at
conversion time — it is read directly from each episode's `meta.json`.

```bash
# Convert all episodes — task read from meta.json (set at recording time):
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings \
    --output_dir ./lerobot_dataset \
    --fps 20

# Override or supply task explicitly (required for older recordings without meta.json task):
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings \
    --output_dir ./lerobot_dataset \
    --task "Pick up cube and place it in the bin" \
    --fps 20

# Without images (faster; suitable when recorded with --no_images):
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings \
    --output_dir ./lerobot_dataset \
    --no_images

# Append a second session to an existing dataset (episodes are numbered sequentially):
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings_session2 \
    --output_dir ./lerobot_dataset \
    --append
```

Use default auto (now prefers q_target_cmd when available):
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
  --input_dir ./recordings \
  --output_dir ./lerobot_dataset_2.4 \
  --fps 20
Force q_target_cmd explicitly:
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
  --input_dir ./recordings \
  --output_dir ./lerobot_dataset_2.4 \
  --fps 20 \
  --action_source q_target_cmd

See [Converting to GR00T Training Format](#converting-to-groot-training-format) for the full
field mapping, options table, and notes on backward compatibility with older recordings.

---

## Recording Controls

Recording is triggered directly from the PICO headset controller:

| Gesture | Action |
|---------|--------|
| **Left grip + right-controller A** | Start recording a new episode (or stop and save the current one) |
| **Left grip + right-controller B** | Abort and discard the current episode |

The recorder prints status to the terminal: `=== Recording started ===`, frame count every 100
frames, and confirmation when an episode is saved.

---

## Recorder Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | `./recordings` | Root directory for saved episodes |
| `--env_name` | `""` | MuJoCo scene used for this recording (e.g. `pnp_cube`). Saved to `meta.json`; used automatically by the converter and playback script. |
| `--task` | `""` | Language task description (e.g. `"Pick up cube"`). Saved to `meta.json`; used automatically by the converter as `--task` default. |
| `--pose_port` | `5556` | ZMQ port for PICO pose stream |
| `--sonic_port` | `5557` | ZMQ port for SONIC g1_debug stream |
| `--image_port` | `5555` | ZMQ port for simulator camera images |
| `--base_state_port` | `5558` | ZMQ port for ground-truth base state from MuJoCo (`0` = disabled) |
| `--host` | `localhost` | Host for ZMQ connections |
| `--no_images` | off | Skip camera image recording |

---

## Output Format

Each episode is saved as a separate subdirectory named by wall-clock time and sequential index:

```
<output_dir>/<YYYYMMDD_HHMMSS>_ep<NNNN>/
├── pico.npz        -- PICO stream (human motion, VR, controller)
├── sonic.npz       -- SONIC stream (WBC target/measured joints, VR)
├── images/
│   ├── head_camera_left/
│   │   ├── 000000.jpg
│   │   └── ...
│   └── head_camera_right/
│       └── ...
└── meta.json       -- n_frames, duration, field lists, env_name, task
```

### `pico.npz` — shape `[T, ...]` where T = number of pose ticks recorded

| Key | Shape | Content |
|-----|-------|---------|
| `smpl_pose` | `[T, N, 21, 3]` | SMPL body pose — axis-angle per joint (N frames buffered per tick; J=21 joints) |
| `smpl_joints` | `[T, N, 24, 3]` | SMPL joint positions in absolute world space (J=24, joint 0 = pelvis) |
| `body_quat_w` | `[T, N, 4]` | Body root quaternion (w-first) |
| `joint_pos` | `[T, N, 29]` | G1 joint positions from SMPL motion retargeting |
| `joint_vel` | `[T, N, 29]` | G1 joint velocities (zeros in current pico_manager) |
| `frame_index` | `[T, N]` | Frame indices for the N buffered frames |
| `vr_position` | `[T, 9]` | VR 3-point positions: [L-wrist, R-wrist, Neck] × xyz |
| `vr_orientation` | `[T, 12]` | VR 3-point orientations: [L, R, Neck] × wxyz |
| `left_hand_joints` | `[T, 7]` | Left Dex3 hand joint target positions (from trigger mapping) |
| `right_hand_joints` | `[T, 7]` | Right Dex3 hand joint target positions (from trigger mapping) |
| `left_trigger` / `right_trigger` | `[T, 1]` | Controller trigger values |
| `left_grip` / `right_grip` | `[T, 1]` | Controller grip values |
| `pico_dt` | `[T, 1]` | Frame delta time (s) — reciprocal of `pico_fps` |
| `pico_fps` | `[T, 1]` | PICO stream rate (Hz) for that frame |
| `timestamp_realtime` | `[T, 1]` | Wall-clock timestamp (s) |
| `timestamp_monotonic` | `[T, 1]` | Monotonic timestamp (s) |
| `heading_increment` | `[T, 1]` | Yaw accumulator change since last tick (rad) |
| `toggle_data_collection` | `[T, 1]` | Rising-edge signal used to start/stop episode recording |
| `toggle_data_abort` | `[T, 1]` | Rising-edge signal used to abort and discard current episode |
| `navigate_cmd` | `[T, 3]` | Active navigate command `[vx, vy, ω_z]` (m/s, m/s, rad/s) — source controlled by `--navigate_cmd_source` |
| `navigate_cmd_joystick` | `[T, 3]` | Joystick-derived navigate command (always recorded) |
| `navigate_cmd_pelvis` | `[T, 3]` | Foot-tracker-derived navigate command from pelvis velocity (NaN `[3]` when foot trackers unavailable) |
| `base_height_cmd_joystick` | `[T, 1]` | Button-driven pelvis height (m); Y raises (+1 cm), X lowers (−1 cm); range 0.20–0.74 m |
| `base_height_cmd_pelvis` | `[T, 1]` | SMPL pelvis Z in robot frame — operator's absolute pelvis height above Unity world origin (m); NaN when foot trackers unavailable |

> **Note:** `navigate_cmd`, `navigate_cmd_joystick`, `navigate_cmd_pelvis`, `base_height_cmd_joystick`, and `base_height_cmd_pelvis` are present only in recordings made after the pico_manager update that added these fields. Older recordings will not have these keys; the conversion script substitutes zeros / default height (0.74 m) for missing fields.

### `sonic.npz` — shape `[T, ...]`, sampled at pose rate from the g1_debug ZMQ stream

> **Input vs output**: the g1_debug stream publishes *both* the inputs to the SONIC/WBC system
> and the motion reference targets it tracks. The final actuator commands (LowCmd) are sent
> directly from deploy.sh to the robot via Unitree SDK DDS and are **not** captured here.

**Inputs — robot state observations** (measured from the simulator via Unitree SDK bridge):

| Key | Shape | Content |
|-----|-------|---------|
| `body_q_measured` | `[T, 29]` | Current joint positions in MuJoCo order, with `default_angles` offsets added |
| `body_dq_measured` | `[T, 29]` | Current joint velocities in MuJoCo order (from Unitree low-state `dq`) |
| `base_quat_measured` | `[T, 4]` | Base orientation from IMU — quaternion wxyz |
| `base_ang_vel_measured` | `[T, 3]` | Base angular velocity from IMU gyroscope — `[wx, wy, wz]` |
| `base_trans_measured` | `[T, 3]` | Base translation; fixed sim default `[0, −1, 0.793]` (not from odometry) |
| `left_hand_q_measured` | `[T, 7]` | Left Dex3 hand joint positions |
| `right_hand_q_measured` | `[T, 7]` | Right Dex3 hand joint positions |

**Inputs — human motion from PICO** (VR controller/headset, passed through from the pose stream):

| Key | Shape | Content |
|-----|-------|---------|
| `vr_3point_position` | `[T, 9]` | Wrist and neck positions rotated into the target body frame — [L-wrist, R-wrist, Neck] × xyz (m) |
| `vr_3point_orientation` | `[T, 12]` | Wrist and neck orientations — [L-wrist, R-wrist, Neck] × wxyz quaternion |
| `vr_3point_compliance` | `[T, 3]` | Per-limb tracking compliance — [L-arm, R-arm, head] |

**Ground-truth simulation state** (requires `--base_state_port 5558` on `run_sim_loop.py` and `record_sonic_teleop.py`):

| Key | Shape | Content |
|-----|-------|---------|
| `base_pos_sim` | `[T, 3]` | Ground-truth base XYZ position from MuJoCo world frame (m) |
| `base_quat_sim` | `[T, 4]` | Ground-truth base quaternion wxyz from MuJoCo |
| `base_linvel_sim` | `[T, 3]` | Ground-truth base linear velocity from MuJoCo world frame (m/s) |
| `base_angvel_sim` | `[T, 3]` | Ground-truth base angular velocity from MuJoCo world frame (rad/s) |

**Reference targets** — joint positions and base pose derived from SMPL motion retargeting,
heading-corrected. These are the targets the WBC control loop is commanded to track;
they are **inputs to SONIC**, not its output:

| Key | Shape | Content |
|-----|-------|---------|
| `body_q_target` | `[T, 29]` | Reference joint positions from SMPL retargeting, MuJoCo order |
| `base_trans_target` | `[T, 3]` | Reference base translation (heading-corrected, m) |
| `base_quat_target` | `[T, 4]` | Reference base quaternion (heading-corrected, wxyz) |

> **Note**: the SONIC network output (LowCmd — actual PD joint targets sent to actuators) is
> transmitted over Unitree SDK `rt/lowcmd` and is not included in this recording. `body_q_target`
> is the closest available proxy: if SONIC tracks the reference perfectly,
> `body_q_target[t] ≈ body_q_measured[t+1]`.

### `images/`

JPEG files at simulator camera rate, organized by camera name and frame index.
Images are stored as-is from the simulator (quality 80 JPEG, 640×480). The frame index in the
filename corresponds to the pose tick at which that image was latest available.

---

## Converting to GR00T Training Format

GR00T N1.5/N1.6 training consumes **LeRobot** datasets (HuggingFace Parquet + H.264 MP4), not raw NPZ files. Use `convert_sonic_to_lerobot.py` to convert.

### Usage

The script requires `lerobot`, `av`, and `pyarrow`, which are available in the `sonic_dc` conda environment.

**Convert all episodes in a directory (task from meta.json):**
```bash
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings \
    --output_dir ./lerobot_dataset \
    --fps 20
```

**Override or supply task explicitly (older recordings without meta.json task):**
```bash
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings \
    --output_dir ./lerobot_dataset \
    --task "Pick up cube and place it in the bin" \
    --fps 20
```

**Convert a single episode directory:**
```bash
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings/20260401_115515_ep0002 \
    --output_dir ./lerobot_dataset
```

**Skip image encoding (faster, smaller output):**
```bash
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings \
    --output_dir ./lerobot_dataset \
    --no_images
```

**Append a second session to an existing dataset:**
```bash
conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \
    --input_dir ./recordings_session2 \
    --output_dir ./lerobot_dataset \
    --append
```

### Conversion options

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | *(required)* | Recording directory (or a single episode dir) |
| `--output_dir` | *(required)* | LeRobot dataset root (created if absent) |
| `--task` | *(from meta.json)* | Language task description (written to `tasks.jsonl`). Optional if the episode was recorded with `record_sonic_teleop.py --task`; required for older recordings. |
| `--fps` | `20` | Output frame rate after downsampling from ~50 Hz PICO rate |
| `--no_images` | off | Skip H.264 video encoding even if `images/` dirs are present |
| `--append` | off | Append to an existing dataset rather than creating a new one |
| `--robot_type` | `g1` | Robot type string written to `meta/info.json` |

### What the script assembles

The SONIC 29-DOF body joint order in `body_q_measured`/`body_q_target` is:
`[left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]`

The 43-DOF layout expected by GR00T training inserts left and right hands after their
respective arms: `[left_leg(6), right_leg(6), waist(3), left_arm(7), left_hand(7), right_arm(7), right_hand(7)]`

| Training field | Shape | Source |
|---|---|---|
| `observation.state` | `[T, 43]` | `body_q_meas[0:22] ‖ lh_meas[7] ‖ body_q_meas[22:29] ‖ rh_meas[7]` |
| `action` | `[T, 43]` | `body_q_tgt[0:22] ‖ pico.left_hand_joints[7] ‖ body_q_tgt[22:29] ‖ pico.right_hand_joints[7]` |
| `observation.eef_state` / `action.eef` | `[T, 14]` | `vr_3pt_pos[0:6] ‖ vr_3pt_ori[0:8]` (L+R wrist, neck dropped) |
| `teleop.navigate_command` | `[T, 3]` | `pico: navigate_cmd` (zeros if not in recording) |
| `teleop.base_height_command` | `[T, 1]` | `pico: base_height_cmd_joystick` (0.74 m default if not in recording) |
| `robot.base_pos` | `[T, 3]` | `sonic: base_pos_sim` — ground-truth pelvis XYZ from MuJoCo world frame (m). Requires `--base_state_port` at recording time. |
| `robot.base_quat` | `[T, 4]` | `sonic: base_quat_sim` — ground-truth pelvis quaternion wxyz from MuJoCo. Requires `--base_state_port` at recording time. |
| `observation.images.*` | video | `images/` JPEGs → H.264 MP4 at `--fps` |
| `task_index` | `[T, 1]` | From `--task` (CLI or `meta.json`); written to `meta/tasks.jsonl` |

Timestamps are resampled from the variable PICO rate (~50 Hz) to the target `--fps` using nearest-neighbour interpolation on `pico.npz` `timestamp_realtime`.

### Key differences from decoupled_wbc collection

| Aspect | decoupled_wbc (gr00t_wbc) | SONIC (gear_sonic) |
|---|---|---|
| Output format | Parquet + H.264 MP4 | NPZ + JPEG → convert with this script |
| Ready for training | Yes — written during collection | No — one conversion pass required |
| Collection rate | 20 Hz | ~50 Hz (PICO); downsampled at conversion |
| Language annotation | Saved at collection time | Supplied via `--task` at conversion time |
| DOF layout | 43-DOF concatenated | 29-body + 7+7 hands reassembled by the script |

---

## Timestamp Sources and Synchronization

The three streams are produced by independent processes and use different timestamp sources:

| Field | Clock | Machine | Notes |
|-------|-------|---------|-------|
| `pico.npz` → `timestamp_realtime` | `time.time()` | Linux workstation | Recorded when PICO SDK returns new body data, before SMPL processing |
| `pico.npz` → `timestamp_monotonic` | `time.monotonic()` | Linux workstation | Same instant as `timestamp_realtime`; unaffected by NTP adjustments |
| `pico.npz` → `timestamp_ns` | PICO device hardware clock | PICO headset (XRoboToolkit SDK) | **Different clock** from system time; used internally for rate control |
| `images/` timestamps | `time.time()` | Linux workstation (image subprocess) | Recorded when rendered frame is written to shared memory |

**`timestamp_realtime` and the image timestamps are both `time.time()` on the same machine** — they share one clock and are directly comparable without conversion. `timestamp_ns` is from the PICO headset's internal clock and has no guaranteed relationship to the Linux system clock.

### Systematic Latency

Even though `timestamp_realtime` and image timestamps share a clock, they capture different events in the pipeline:

```
PICO device data ready
  │  xrt.get_time_stamp_ns() → new stamp
  │  time.time()  ← timestamp_realtime recorded here
  ▼
  SMPL inference + ZMQ pack + send  (~5–20 ms)
  ▼
  [recorder receives pose message; samples image_holder.get()]

  sim physics step (200 Hz)
  ▼
  offscreen render (every 6 steps → 30 Hz)
  ▼
  shared memory write → data_ready_event fires
  │  time.time()  ← image timestamp recorded here
  ▼
  JPEG encode + ZMQ send
```

The image timestamp is recorded **after** the frame is rendered (~1–5 ms JPEG encoding latency),
and `timestamp_realtime` is recorded **before** SMPL processing and ZMQ send (~5–20 ms pipeline
latency). The net cross-stream offset is **10–30 ms**, systematic and consistent across a session.

For most training use cases this offset is acceptable. If sub-frame alignment is required, three
calibration options are available:

**Option 1 — Empirical clap calibration (simplest)**

Clap hands sharply in front of the camera once at the start of each session. Find the JPEG frame
where hands meet, and find the `timestamp_realtime` row where hand-contact motion occurs.
Their difference is the calibration offset for that session.

**Option 2 — Pipeline latency logging**

Record `time.time()` immediately before `socket.send()` in pico_manager (line 1469 of
`pico_manager_thread_server.py`) and immediately after `pose_sock.recv()` in the recorder. These
two extra timestamps bracket the ZMQ transmission and processing latency end-to-end.

**Option 3 — Calibrate `timestamp_ns` against system time**

Record both clocks at startup to compute a fixed epoch offset:

```python
t_system  = time.time()
t_pico_ns = xrt.get_time_stamp_ns()
pico_epoch_offset_s = t_system - t_pico_ns * 1e-9
# Convert any later timestamp_ns to wall clock:
t_wall = pico_epoch_offset_s + timestamp_ns * 1e-9
```

This lets you use the PICO hardware clock (which increments at a stable rate independent of NTP)
as the synchronization anchor across all three streams.

---

## Visualizing Recorded Episodes

`playback_episode.py` converts a saved episode directory into a single MP4 for review. Each frame is laid out as a row of panels:

```
┌──────────────────┬──────────────────┬──────────────────┐
│  head_cam_left   │  head_cam_right  │  SMPL skeleton   │
│  (+ HUD overlay) │                  │  (3-D view)      │
└──────────────────┴──────────────────┴──────────────────┘
```

The HUD on the leftmost panel shows frame index, elapsed time, measured/target knee and elbow angles, and VR wrist positions.

### Usage

```bash
source .venv_teleop/bin/activate

# Write to <episode_dir>/playback.mp4 at the recorded fps
python gear_sonic/scripts/playback_episode.py <episode_dir>

# Custom output path and fps
python gear_sonic/scripts/playback_episode.py <episode_dir> \
    --output /tmp/demo.mp4 --fps 25

# Skip SMPL skeleton panel (faster, no matplotlib rendering)
python gear_sonic/scripts/playback_episode.py <episode_dir> --no_skeleton
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `episode_dir` | *(required)* | Path to a directory saved by `record_sonic_teleop.py` |
| `--output` | `<episode_dir>/playback.mp4` | Output video path |
| `--fps` | inferred from timestamps | Video frame rate |
| `--no_skeleton` | off | Omit the 3-D SMPL skeleton panel |
| `--panel_height` | `480` | Height of each panel in pixels |

The output is H.264 / yuv420p (re-encoded with ffmpeg for broad player compatibility). `ffmpeg` must be available on `PATH`.

---

## Replaying LeRobot Episodes in MuJoCo

`playback_lerobot.py` kinematically replays a converted LeRobot episode directly in the MuJoCo
simulator — no C++ WBC process or Unitree SDK needed. It reads `observation.state` (43-DOF joint
positions) and `robot.base_pos` / `robot.base_quat` (root pose) from the Parquet file, sets
`qpos` directly, and calls `mj_forward()` each frame. An optional video is rendered from any
named camera in the scene.

```{admonition} Root pose required for correct lower-body replay
:class: note
Without `robot.base_pos` / `robot.base_quat` the robot is fixed at a standing position and
any locomotion during recording (including balance-stepping while stationary) will not be
replayed — only arm movements will be visible. These columns are present when episodes are
recorded with `--base_state_port` (the default).
```

### Usage

```bash
# Replay episode 0 with the MuJoCo viewer + save an MP4 (use the correct --env_name):
conda run -n sonic_dc python gear_sonic/scripts/playback_lerobot.py \
    --dataset_dir ./lerobot_dataset \
    --episode 0 \
    --env_name pnp_cube \
    --output_video playback_ep0.mp4

# Headless — video only, no GUI window:
conda run -n sonic_dc python gear_sonic/scripts/playback_lerobot.py \
    --dataset_dir ./lerobot_dataset \
    --episode 0 \
    --env_name pnp_cube \
    --output_video playback_ep0.mp4 \
    --no_viewer

# Playback with policy
cd /home/horizon/wrk/SONIC/GR00T-WholeBodyControl
source .venv_teleop/bin/activate

python gear_sonic/scripts/playback_lerobot.py \
  --dataset_dir ./lerobot_dataset_2.2 \
  --episode 0 \
  --env_name pnp_cube \
  --fps 20 \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --compare \
  --output_video ./playback_policy_ep0.mp4 \
  --camera overview

# playback with policy and physics
python gear_sonic/scripts/playback_lerobot.py \
  --dataset_dir ./lerobot_dataset_2.2 \
  --episode 0 \
  --env_name pnp_cube \
  --fps 20 \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --compare \
  --physics \
  --output_video ./playback_policy_ep0_physics.mp4 \
  --camera overview

# Playback using recorded encoder/decoder history inputs (no 10-frame reconstruction in replay):
# - Encoder uses recorded SMPL-mode blocks from sonic.encoder_obs:
#     smpl_joints_10frame_step1 / smpl_anchor_orientation_10frame_step1 /
#     motion_joint_positions_wrists_10frame_step1
# - Decoder uses recorded sonic.decoder_obs directly (his_* 10-frame inputs from recording).
# - Decoder output is still recomputed by ONNX (not copied from recorded decoder_action_raw/q_target_cmd).
# - Writes a per-frame model-I/O comparison report txt (default):
#     <dataset_dir>/episode_<episode>_model_io_frame_compare.txt
#   The same txt now includes:
#     * decoder_action_raw replay vs sonic.decoder_action_raw recorded (full vectors, each frame)
#     * q_target_cmd replay vs sonic.q_target_cmd recorded (full vectors + per-frame L2)
python gear_sonic/scripts/playback_lerobot_recorded_smpl_blocks.py \
  --dataset_dir ./lerobot_dataset_2.4 \
  --episode 0 \
  --env_name pnp_cube \
  --fps 20 \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --compare \
  --output_video ./playback_policy_ep0_recorded_blocks.mp4 \
  --camera overview

# Optional: set custom report path
python gear_sonic/scripts/playback_lerobot_recorded_smpl_blocks.py \
  --dataset_dir ./lerobot_dataset_2.4 \
  --episode 0 \
  --env_name pnp_cube \
  --fps 20 \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --indicator_report_txt ./debug/ep0_model_io_compare.txt
```
python gear_sonic/scripts/playback_lerobot_recorded_smpl_blocks.py \
  --dataset_dir ./lerobot_dataset_2.4 \
  --episode 0 \
  --env_name pnp_cube \
  --fps 20 \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --compare \
  --force_frame0_recorded_state \
  --force_recorded_prefix_frames 11 \
  --output_video ./playback_policy_ep0_recorded_blocks.mp4 \
  --camera head_camera

playback action
python gear_sonic/scripts/playback_lerobot_action.py   
--dataset_dir ./lerobot_dataset_2.4   --episode 0   --env_name pnp_cube   --fps 20   --output_video ./playback_action_ep0.mp4   --camera overview 

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_dir` | *(required)* | LeRobot dataset root (must contain `data/` and `meta/`) |
| `--episode` | `0` | Episode index to replay |
| `--env_name` | `kitchen_pnp_apple` | MuJoCo scene to load. **Must match the scene used during recording.** |
| `--output_video` | *(none)* | Path to save an MP4 (e.g. `playback_ep0.mp4`). Omit to skip. |
| `--camera` | `overview` | Camera name for video rendering. Falls back to first available camera if not found. |
| `--no_viewer` | off | Disable the interactive MuJoCo viewer window (headless rendering) |
| `--fps` | `20` | Playback and video frame rate |
| `--video_width` | `1280` | Video width in pixels |
| `--video_height` | `720` | Video height in pixels |

```{admonition} --env_name must match the recording
:class: warning
`--env_name` selects the MuJoCo XML scene used for replay. If it does not match the scene
that was active during recording, the robot will be in the wrong environment (wrong objects,
wrong camera positions). When episodes are recorded with `record_sonic_teleop.py --env_name`,
the correct value is stored in `meta.json` of each episode directory.
```

---

## Customizing Simulation Scenes

The simulator environment is selected with `--env_name` on `run_sim_loop.py`. Each environment
name maps to a Python class that points to a MuJoCo XML scene file.

### Built-in environments

| `--env_name` | Task | XML scene |
|---|---|---|
| `default` | Empty room, robot only | `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml` |
| `pnp_cube` | Pick and place a cube | `decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml` |
| `lift_box` | Bimanual box lift | `decoupled_wbc/control/robot_model/model_data/g1/lift_box_43dof.xml` |
| `pnp_bottle` | Pick and place a bottle | `decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml` |
| `kitchen_pnp_apple` | Pick apple → place on plate (kitchen) | `decoupled_wbc/control/robot_model/model_data/g1/kitchen_pnp_apple_43dof.xml` |

### Running with a built-in scene

Replace `--env_name` with whichever environment you want:

```bash
# Cube pick-and-place scene, with stereo head cameras
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
    --env_name pnp_cube \
    --head_cam \
    --enable_image_publish \
    --enable_offscreen
```

All other terminals (WBC, PICO manager, recorder) are started exactly as described in
[Setup](#setup-5-terminals) — only the `--env_name` argument to `run_sim_loop.py` changes.

### Creating a new scene

Adding a custom scene requires three steps.

#### Step 1 — Write the MuJoCo XML

Create a new XML file in `decoupled_wbc/control/robot_model/model_data/g1/`. Start from an
existing scene and modify `<worldbody>` to add your objects.

The mandatory first line must include the robot model:

```xml
<mujoco model="my_scene">
  <include file="g1_29dof_with_hand_rev_1_0_activatedfinger.xml" />

  <asset>
    <material name="my_obj_mat" rgba="0.8 0.2 0.1 1" />
  </asset>

  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" rgba="0.8 0.8 0.8 1" />

    <!-- Fixed furniture (no joint) -->
    <body name="table_body" pos="1.2 0 0">
      <geom name="table_top" pos="0 0 0.85" size="0.35 0.6 0.05" type="box" rgba="0.7 0.6 0.5 1" />
      <geom name="table_base" pos="0 0 0.4"  size="0.35 0.6 0.40" type="box" rgba="0.6 0.5 0.4 1" />
    </body>

    <!-- Dynamic object (free joint so it can be picked up) -->
    <body name="my_object_body" pos="1.1 0 0.94">
      <joint type="free" damping="0.0008" name="my_object_joint" />
      <geom name="my_object" type="sphere" size="0.04" material="my_obj_mat"
        solimp="0.998 0.998 0.001" solref="0.001 2" density="120" friction="0.95 0.35 0.10" />
    </body>
  </worldbody>

  <default>
    <geom friction="1.0" />
  </default>
</mujoco>
```

**Key rules:**

- `<include file="..."/>` path is relative to the XML file's own directory.
- Do **not** redeclare cameras named `head_camera`, `head_camera_left`, or `head_camera_right` —
  they are already defined inside the included robot XML.
- Dynamic objects (things the robot picks up) need `<joint type="free" .../>`.
- Fixed scene geometry (tables, walls, appliances) needs no joint.
- All box geoms require **three** size values (half-widths in x, y, z). Cylinder geoms require
  **two** (radius, half-height). Using the wrong number causes a MuJoCo load error.
- Visual-only decorations (handles, labels, etc.) that should not affect physics:
  set `contype="0" conaffinity="0"`.

#### Step 2 — Add a Python environment class

Open `gear_sonic/utils/mujoco_sim/base_sim.py` and add a class after `BottleEnv`:

```python
class MySceneEnv(DefaultEnv):
    """One-line description of the task."""

    def __init__(self, config: Dict[str, any], **kwargs):
        config = config.copy()
        config["ROBOT_SCENE"] = (
            "decoupled_wbc/control/robot_model/model_data/g1/my_scene_43dof.xml"
        )
        super().__init__(config, "my_scene", **kwargs)

    def update_reward(self):
        """Return True when the task is complete (called at ~50 Hz)."""
        success = check_contact(self.mj_model, self.mj_data, "my_object_body", "target_body")
        with self.reward_lock:
            self.last_reward = success
```

`check_contact(model, data, body_a, body_b)` returns `True` when any geom of `body_a` touches any
geom of `body_b`. `check_height(model, data, geom_name, z_low, z_high)` checks whether a geom's
centre is within a height range — useful for lifted-object conditions.

#### Step 3 — Register the environment name

In the same file, find the `if / elif` block inside `BaseSimulator.__init__` and add a branch:

```python
elif env_name == "my_scene":
    self.sim_env = MySceneEnv(config, **kwargs)
```

Also extend the error message in the `else` branch so the new name appears in validation output.

#### Step 4 — Launch and record

```bash
# Terminal 1 — simulator with the new scene
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
    --env_name my_scene \
    --head_cam \
    --enable_image_publish \
    --enable_offscreen \
    --base_state_port 5558

# Terminal 5 — recorder (unchanged)
python gear_sonic/scripts/record_sonic_teleop.py \
    --output_dir ./recordings/my_scene
```

### Worked example — kitchen pick-and-place (apple → plate)

The `kitchen_pnp_apple` environment illustrates a complete custom scene:

**Scene** (`kitchen_pnp_apple_43dof.xml`):
- Fully enclosed room: 2.6 m deep × 4.0 m wide × 2.4 m tall with tile floor, four walls, ceiling,
  door opening (front wall), and a window with a frame (right wall).
- L-shaped kitchen counter running along the back wall: base cabinet, marble-look worktop
  (surface at z = 0.90 m), backsplash tile panel, wall cabinets above.
- Appliances on the counter: microwave (right), toaster (left), stainless-steel sink with faucet (centre-left).
- **Apple** — red sphere (r = 40 mm), free joint, placed at y = −0.20 m (left of centre on counter).
- **Plate** — white cylinder (r = 115 mm, h = 20 mm), fixed, placed at y = +0.25 m (right of centre).

**Object geometry reference:**

| Object | Body name | Geom name | Type | Key size |
|---|---|---|---|---|
| Apple | `apple_body` | `apple` | sphere | r = 0.040 m |
| Plate | `plate_body` | `plate` | cylinder | r = 0.115 m, h = 0.010 m |

**Success condition** (`KitchenAppleToPlateEnv.update_reward`):

```python
apple_on_plate      = check_contact(model, data, "apple_body", "plate_body")
apple_at_plate_height = check_height(model, data, "apple", 0.945, 1.05)
reward = apple_on_plate & apple_at_plate_height
```

Apple resting on the counter sits at z ≈ 0.940 (below the 0.945 threshold), so the reward is
`False` until the apple is lifted onto the plate (z ≈ 0.960).

**Launch commands:**

```bash
# Terminal 1 — kitchen simulator
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
    --env_name kitchen_pnp_apple \
    --head_cam \
    --enable_image_publish \
    --enable_offscreen \
    --base_state_port 5558

# Terminal 2 — WBC (unchanged)
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim --input-type zmq_manager

# Terminal 3 — PICO manager (POSE mode, pelvis navigate_cmd)
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --waist_tracking --vis_vr3pt

# Terminal 4 — headset video stream (optional)
source .venv_teleop/bin/activate
python gear_sonic/scripts/stream_cam_xr.py

# Terminal 5 — recorder
source .venv_teleop/bin/activate
python gear_sonic/scripts/record_sonic_teleop.py \
    --output_dir ./recordings/kitchen_pnp_apple
```

To collect without camera images (faster, smaller files):

```bash
python gear_sonic/scripts/record_sonic_teleop.py \
    --output_dir ./recordings/kitchen_pnp_apple \
    --no_images
```

---

## Tips for Quality Data

1. **Wear tight-fitting clothing** — Required for reliable foot tracker visibility (see [Teleoperation Guide](teleoperation.md#clothing-requirements))
2. **Recalibrate often** — PICO tracking drifts over time; recalibrate between episodes
3. **Discard jittery episodes** — Use **Left grip + right-controller B** immediately if you see tracking glitches
4. **Keep WiFi latency < 10ms** — High latency causes unnatural robot motion and poor training data
5. **Consistent task framing** — Position objects in the same region across episodes for better policy generalization
6. **Short, clean episodes** — Complete the task cleanly; avoid lingering or correcting mid-episode

---

## Related Documentation

- [Whole-body Teleoperation Guide](teleoperation.md) — setup, safety, movement practices
- [VR Teleop Setup](../getting_started/vr_teleop_setup.md) — PICO hardware setup
- [Troubleshooting](troubleshooting.md) — common issues and fixes
