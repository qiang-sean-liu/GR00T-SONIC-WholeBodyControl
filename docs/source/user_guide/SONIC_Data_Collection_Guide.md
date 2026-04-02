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

Recording is controlled directly from the PICO headset — no keyboard interaction needed.

---

## Setup (5 terminals)

Follow the [Teleoperation Guide](teleoperation.md#running-mujoco-teleop) to bring up Terminals 1–4,
then add Terminal 5 for the recorder.

**Terminal 1 — MuJoCo Simulator** (with image publishing enabled):
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
    --env_name pnp_cube \
    --head_cam \
    --enable_image_publish \
    --enable_offscreen
```

`--head_cam` renders stereo head cameras; `--enable_image_publish` publishes them over ZMQ on
port 5555. Without these flags, no camera images are saved.

**Terminal 2 — C++ WBC + SONIC**:
```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim --input-type zmq_manager
```

**Terminal 3 — PICO Manager**:
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --waist_tracking --vis_vr3pt
```

**Terminal 4 — Headset Video Stream** (optional — stream head cam to PICO):
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/stream_cam_xr.py
```

**Terminal 5 — Recorder**:
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/record_sonic_teleop.py \
    --output_dir ./recordings
```

To skip camera images (faster, smaller files):
```bash
python gear_sonic/scripts/record_sonic_teleop.py --output_dir ./recordings --no_images
```

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
| `--pose_port` | `5556` | ZMQ port for PICO pose stream |
| `--sonic_port` | `5557` | ZMQ port for SONIC g1_debug stream |
| `--image_port` | `5555` | ZMQ port for simulator camera images |
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
└── meta.json       -- n_frames, duration, timestamps, field lists
```

### `pico.npz` — shape `[T, ...]` where T = number of pose ticks recorded

| Key | Shape | Content |
|-----|-------|---------|
| `smpl_pose` | `[T, N, 72]` | SMPL body pose params (N frames buffered per tick) |
| `smpl_joints` | `[T, N, J, 3]` | SMPL joint positions |
| `body_quat_w` | `[T, N, 4]` | Body root quaternion (w-first) |
| `joint_pos` | `[T, N, 29]` | G1 joint positions from motion retargeting |
| `joint_vel` | `[T, N, 29]` | G1 joint velocities (zeros in current pico_manager) |
| `vr_position` | `[T, 9]` | VR 3-point positions: [L-wrist, R-wrist, Neck] × xyz |
| `vr_orientation` | `[T, 12]` | VR 3-point orientations: [L, R, Neck] × wxyz |
| `left_hand_joints` | `[T, 7]` | Left Dex3 hand joint positions |
| `right_hand_joints` | `[T, 7]` | Right Dex3 hand joint positions |
| `left_trigger` / `right_trigger` | `[T, 1]` | Controller trigger values |
| `left_grip` / `right_grip` | `[T, 1]` | Controller grip values |
| `timestamp_realtime` | `[T, 1]` | Wall-clock timestamp (s) |
| `timestamp_monotonic` | `[T, 1]` | Monotonic timestamp (s) |
| `heading_increment` | `[T, 1]` | Yaw accumulator change since last tick (rad) |

### `sonic.npz` — shape `[T, ...]`, sampled at pose rate from the g1_debug ZMQ stream

> **Input vs output**: the g1_debug stream publishes *both* the inputs to the SONIC/WBC system
> and the motion reference targets it tracks. The final actuator commands (LowCmd) are sent
> directly from deploy.sh to the robot via Unitree SDK DDS and are **not** captured here.

**Inputs — robot state observations** (measured from the simulator via Unitree SDK bridge):

| Key | Shape | Content |
|-----|-------|---------|
| `body_q_measured` | `[T, 29]` | Current joint positions in MuJoCo order, with `default_angles` offsets added |
| `base_quat_measured` | `[T, 4]` | Base orientation from IMU — quaternion wxyz |
| `base_trans_measured` | `[T, 3]` | Base translation; fixed sim default `[0, −1, 0.793]` (not from odometry) |
| `left_hand_q_measured` | `[T, 7]` | Left Dex3 hand joint positions |
| `right_hand_q_measured` | `[T, 7]` | Right Dex3 hand joint positions |

**Inputs — human motion from PICO** (VR controller/headset, passed through from the pose stream):

| Key | Shape | Content |
|-----|-------|---------|
| `vr_3point_position` | `[T, 9]` | Wrist and neck positions rotated into the target body frame — [L-wrist, R-wrist, Neck] × xyz (m) |
| `vr_3point_orientation` | `[T, 12]` | Wrist and neck orientations — [L-wrist, R-wrist, Neck] × wxyz quaternion |
| `vr_3point_compliance` | `[T, 3]` | Per-limb tracking compliance — [L-arm, R-arm, head] |

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
