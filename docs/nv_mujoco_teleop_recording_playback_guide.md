# Branch Guide: Data Recording, LowCmd Debug, And ONNX Playback

This guide documents the current branch workflow for SONIC MuJoCo teleoperation data recording and playback debugging. It focuses on the five-terminal recording setup, the new debug datasets, the new scripts added on this branch, and the main findings from the Episode 3 ONNX playback investigation.

## 1. Five-Terminal Data Recording Startup

Run all commands from the repository root unless the command explicitly changes directory:

```bash
cd /home/horizon/wrk/SONIC/GR00T-WholeBodyControl
```

### Terminal 1: Start MuJoCo

For standard simulation data collection with camera publishing:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
  --enable-image-publish \
  --enable-offscreen \
  --camera-port 5555
```

For the current table / object setup, use the PICO bringup wrapper if the local `pnp_cube` XML/head-camera scene is needed:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop_pico_bringup.py \
  --env_name pnp_cube \
  --head_cam \
  --enable_image_publish \
  --enable_offscreen \
  --camera-port 5555
```

What this starts:

- MuJoCo physics at `SIMULATE_DT = 0.005`, i.e. 200 Hz.
- DDS bridge for `LowState` / `LowCmd`.
- Base-state debug publisher on ZMQ port `5558`.
- Camera image publisher on ZMQ port `5555` when image publishing is enabled.

Important details:

- `--enable-image-publish` requires `--enable-offscreen`.
- The data exporter and camera viewer expect the camera server at `localhost:5555`.
- The lowcmd / substep exporters expect the MuJoCo base-state debug stream at `localhost:5558`.

### Terminal 2: Load The C++ Policy

For simulation:

```bash
cd gear_sonic_deploy
./deploy.sh --input-type manager --output-type all sim
```

Equivalent command seen in earlier runs:

```bash
cd gear_sonic_deploy
./deploy.sh sim --input-type zmq_manager
```

The current `deploy.sh` defaults are:

```text
checkpoint:  policy/release/model
decoder:     policy/release/model_decoder.onnx
encoder:     policy/release/model_encoder.onnx
obs config:  policy/release/observation_config.yaml
planner:     planner/target_vel/V2/planner_sonic.onnx
input type:  manager
output type: all
```

What this starts:

- The 50 Hz C++ control loop.
- TensorRT policy inference through `PolicyEngine`.
- The 500 Hz `LowCommandWriter`, which publishes `LowCmd`.
- ZMQ debug output on port `5557`, including `g1_debug` and periodic `robot_config`.
- ONNX debug fields on this branch: `encoder_obs`, `token_state`, `decoder_obs`, `decoder_action_raw`, and `q_target_cmd`.

Operator action:

- `deploy.sh` may wait for confirmation. Press Enter in the deploy terminal when ready.

### Terminal 3: Load Calibration / PICO Teleop Streamer

Use the PICO manager streamer for calibration and teleoperation input:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --waist_tracking \
  --vis_vr3pt \
  --vis_smpl
```

What this provides:

- PICO / body tracking stream to the C++ manager input.
- Calibration modes used by the deploy side.
- Optional visualizations for VR 3-point and SMPL streams.

Operator calibration flow:

- Wear the PICO headset, controllers, and trackers.
- Stand in the calibration pose.
- Press `A+B+X+Y` to start policy / full calibration.
- Press `A+X` to enter whole-body POSE mode.
- Recalibrate whenever tracking quality drifts.

### Terminal 4: Start Data Recording

There are three relevant exporter levels on this branch.

Original SONIC data recording:

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "pick up the object" \
  --data-collection-frequency 50 \
  --camera-host localhost \
  --camera-port 5555
```

LowCmd / MuJoCo debug recording:

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter_lowcmd.py \
  --task-prompt "pick up the object" \
  --data-collection-frequency 50 \
  --camera-host localhost \
  --camera-port 5555 \
  --state-zmq-host localhost \
  --state-zmq-port 5557 \
  --base-state-zmq-host localhost \
  --base-state-zmq-port 5558
```

ONNX inputs + LowCmd + MuJoCo substep debug recording:

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter_onnx_inputs.py \
  --dataset-name onnx-inputs-debug \
  --task-prompt "pick up the object" \
  --data-collection-frequency 50 \
  --camera-host localhost \
  --camera-port 5555 \
  --state-zmq-host localhost \
  --state-zmq-port 5557 \
  --base-state-zmq-host localhost \
  --base-state-zmq-port 5558
```

What to use:

- Use `run_data_exporter.py` for ordinary SONIC dataset recording.
- Use `run_data_exporter_lowcmd.py` when you need `LowCmd`, body/hand command fields, MuJoCo qpos/qvel, and timing debug.
- Use `run_data_exporter_onnx_inputs.py` when you need exact C++ SONIC encoder / decoder inputs and outputs for ONNX playback diagnosis.

### Terminal 5: Camera View

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host localhost \
  --camera-port 5555
```

Controls:

- `R`: start / stop camera recording.
- `Q`: quit viewer.

The viewer is only for visual monitoring / optional camera MP4 recording. Dataset image recording is handled by the data exporter.

### Optional: One-Command tmux Launcher

The script `gear_sonic/scripts/launch_data_collection.py` can launch the same components in a tmux session. It starts MuJoCo, C++ deploy, PICO teleop, data exporter, and optionally the camera viewer. Use it when you want a reproducible pane layout, but for debugging this branch the manual five-terminal setup is easier to inspect.

## 2. Findings From The Branch Investigations

Detailed source docs:

- `docs/onnx_inputs_playback_ep3_comparison.md`
- `docs/onnx_inputs_ep3_substep_q_des_analysis.md`
- `docs/onnx_inputs_ep3_motor_pd_q_des_analysis.md`
- `docs/onnx_inputs_ep3_motor_pd_substep_q_des_joint10_right_ankle_pitch.md`
- `docs/onnx_inputs_ep3_motor_pd_substep_q_and_q_des_joint10_right_ankle_pitch.md`

### What Is Actually Recorded At Each MuJoCo Substep

`gear_sonic/utils/mujoco_sim/base_sim.py` records a pre-step snapshot immediately before every `mj_step`, then appends post-step results after `mj_step`.

Pre-step fields include:

- `robot.motor_pd_substep_q`
- `robot.motor_pd_substep_dq`
- `robot.motor_pd_substep_q_des`
- `robot.motor_pd_substep_dq_des`
- `robot.motor_pd_substep_kp`
- `robot.motor_pd_substep_kd`
- `robot.motor_pd_substep_tau_ff`
- `robot.motor_pd_substep_torque_raw`
- `robot.motor_pd_substep_torque`
- `robot.motor_pd_substep_sim_time`
- `robot.mujoco_substep_qpos`
- `robot.mujoco_substep_qvel`
- `robot.mujoco_substep_ctrl`
- `robot.mujoco_substep_qfrc_applied`
- `robot.mujoco_substep_xfrc_applied`
- `robot.mujoco_substep_qacc_warmstart`

Post-step fields include:

- `robot.mujoco_substep_post_qpos`
- `robot.mujoco_substep_post_qvel`
- `robot.mujoco_substep_post_qacc`
- `robot.mujoco_substep_post_actuator_force`
- `robot.mujoco_substep_post_sim_time`

The exporter does not recompute these values. It stores the arrays received from the MuJoCo base-state debug stream.

### Why There Are Four Substeps Per Frame

The simulator runs at 200 Hz:

```text
SIMULATE_DT = 0.005 s
```

The dataset frame rate is 50 Hz:

```text
dataset dt = 0.02 s
```

So:

```text
0.02 / 0.005 = 4 substeps
```

Important: the four substeps are a retained debug history window, not an atomic action window. The simulator does not freeze one policy command for all four substeps.

### Why `q_des` Changes Inside One Frame

The C++ policy updates `motor_command_buffer_` at 50 Hz in `CreatePolicyCommand()`.

The C++ `LowCommandWriter` publishes the latest command at 500 Hz.

The Python MuJoCo side receives `LowCmd` asynchronously:

```python
def LowCmdHandler(self, msg):
    with self.low_cmd_lock:
        self.low_cmd = msg
        self.low_cmd_received = True
        self.new_low_cmd = True
```

Each 200 Hz MuJoCo `sim_step()` computes torque from the latest stored `LowCmd`:

```python
self.unitree_bridge.low_cmd.motor_cmd[i].q
```

Therefore a new `LowCmd` can arrive between two MuJoCo substeps. In Episode 3:

```text
total frames:                         1125
frames with exactly 4 valid substeps: 1124
frames with constant q_des:           201
frames with changing q_des:           924
frames with 3 q_des values:           6
```

Frames with 3 distinct substep `q_des` values:

```text
91, 246, 298, 612, 922, 997
```

Those rare 3-command frames happen when wall-time jitter makes the retained four MuJoCo substeps span more than one C++ 20 ms policy-command boundary.

### Frame-Level `robot.motor_pd_q_des`

`robot.motor_pd_q_des` is the latest `LowCmd.motor_cmd[i].q` seen at the frame/debug-snapshot level.

In Episode 3:

```text
finite values: 32625 / 32625
robot.motor_pd_q_des matches latest valid substep q_des: 1125 / 1125 frames
```

This means `robot.motor_pd_q_des` is a useful latest-command summary, but it is not sufficient for exact substep replay. Exact replay must use `robot.motor_pd_substep_q_des`.

### `action.wbc` Alignment

`action.wbc` is frame-level and asynchronous relative to the LowCmd / MuJoCo substep stream.

For Episode 3, `action.wbc[t]` body values align best with the previous policy command:

```text
action.wbc[t] body vs sonic.q_target_cmd[t-1]:
  rows_allclose_1e-6 = 1077 / 1124
  median_l2          = 5.036e-08

action.wbc[t] body vs sonic.q_target_cmd[t]:
  rows_allclose_1e-6 = 0 / 1125
  median_l2          = 7.177e-02
```

For command-accurate replay, prefer:

- `sonic.q_target_cmd`
- `robot.motor_pd_q_des`
- `robot.motor_pd_substep_q_des` for exact substep replay

Do not derive body command timing from `action.wbc` unless the offset is explicitly accounted for.

### ONNX Playback Backend Difference

The recorded teleop decoder output came from:

```text
C++ deploy -> PolicyEngine -> TensorRT -> FP32
```

Python playback with `playback_lerobot_onnx_inputs_pd.py` uses:

```text
Python ONNXRuntime CPU FP32
```

Even when the same recorded `sonic.decoder_obs` is fed to the model, ONNXRuntime CPU and TensorRT FP32 are not bit-identical. During warmup, this explains small deterministic differences around `1e-6`.

The isolated C++ proof tool showed exact reproduction:

```text
raw_action_vs_recorded:
  mean_l2 = 0
  max_l2  = 0

q_target_vs_recorded:
  mean_l2 = 0
  max_l2  = 0
```

Conclusion: the Python playback implementation was not missing decoder inputs. The small difference is backend-related.

### Stale ZMQ / Debug Frames

Episode 3 contains 22 frames where:

- `observation.state`
- `sonic.decoder_obs`
- `sonic.decoder_action_raw`

are identical to the previous frame.

The stale frames are:

```text
157, 162, 166, 171, 177, 181, 186, 216, 221, 226, 231,
471, 476, 481, 486, 491, 496, 805, 810, 815, 820, 825
```

In 21 of those frames, `robot.motor_pd_q_des` still changed. This confirms that the policy/debug stream and MuJoCo LowCmd-derived command fields can advance differently.

### `qacc_warmstart`

`qacc_warmstart` is not generated by teleop or the policy. It is an internal MuJoCo solver warm-start state. It depends on previous physics state, contacts, constraints, qpos/qvel, ctrl, and solver history.

For exact substep replay, restore it together with:

```text
data.time
data.ctrl
data.qfrc_applied
data.xfrc_applied
data.qacc_warmstart
```

However, in the Frame 12 investigation, the first meaningful divergence was caused by stale `q_des`, not `qacc_warmstart`.

## 3. New Scripts And Branch Changes

### Data Exporters

`gear_sonic/scripts/run_data_exporter.py`

- Original SONIC exporter.
- Records normal LeRobot fields, camera frames, teleop inputs, and robot/debug state.
- Output naming defaults to timestamp only, for example `2026-04-22-19-16-43`.

`gear_sonic/scripts/run_data_exporter_lowcmd.py`

- Extends the original exporter with LowCmd and MuJoCo debug fields.
- Records body and hand command `q/dq/kp/kd/tau`.
- Records MuJoCo qpos/qvel and timestamps.
- Uses base-state debug messages when available.
- Output naming defaults to `*-lowcmd-debug`.

`gear_sonic/scripts/run_data_exporter_onnx_inputs.py`

- Extends `run_data_exporter_lowcmd.py`.
- Adds exact SONIC ONNX debug buffers:
  - `robot.base_ang_vel`
  - `robot.body_dq`
  - `sonic.encoder_obs`
  - `sonic.token_state`
  - `sonic.decoder_obs`
  - `sonic.decoder_action_raw`
  - `sonic.q_target_cmd`
- Adds legacy MuJoCo substep fields needed for playback comparison.
- Checks the `g1_debug` ZMQ frame rate at startup.
- Output naming defaults to `*-onnx-inputs-debug`, or use `--dataset-name onnx-inputs-debug`.

### Playback And Analysis Scripts

`gear_sonic/scripts/playback_lerobot_action_pd.py`

- Replays recorded data through teleop-style PD control.
- Can use recorded substep inputs with `--use_recorded_substep_inputs`.
- Best script for verifying exact MuJoCo substep replay.

`gear_sonic/scripts/playback_lerobot_onnx_inputs_pd.py`

- Replays recorded ONNX encoder/decoder inputs through Python ONNXRuntime.
- Initializes MuJoCo from recorded data, then lets playback physics roll forward.
- Good for closed-loop ONNX playback comparison.

`gear_sonic/scripts/playback_lerobot_onnx_inputs_pd_decoder_warmup.py`

- Hybrid playback script.
- Uses recorded `sonic.decoder_obs` and recorded MuJoCo substeps for a warmup window, then switches to closed-loop rollout.
- Adds comparison logging for `L2_token`, `L2_q_target`, and state differences.
- Handles recorded hand gains/state fixes for closer visual and physics replay around handoff.

`gear_sonic/scripts/playback_lerobot_onnx_inputs_pd_decoder_warmup_trt.py`

- Wrapper around the warmup playback path using the TensorRT bridge instead of Python ONNXRuntime for inference.

`gear_sonic/scripts/playback_lerobot_onnx_future_inputs_pd.py`

- Reconstructs encoder future-input sections from recorded teleop future fields.
- Verifies that the future sections reproduce recorded `sonic.encoder_obs`.

`gear_sonic/scripts/compare_recorded_decoder_with_policy_engine.py`

- Extracts recorded `sonic.decoder_obs`, `sonic.decoder_action_raw`, and `sonic.q_target_cmd`.
- Runs them through the C++ TensorRT replay proof executable.
- Confirms whether C++ `PolicyEngine` reproduces recorded decoder outputs exactly.

`gear_sonic/scripts/sonic_trt_inference_bridge.py`

- Python client for a persistent C++ TensorRT inference subprocess.
- Provides ONNXRuntime-like `run()` adapters for encoder and decoder playback.

### C++ Debug / Replay Additions

`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`

- Stores latest decoder input, raw decoder action, and final body q target.
- Logs post-policy ONNX debug state through `StateLogger`.

`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/state_logger.hpp`
and `src/state_logger.cpp`

- Add fields for exact encoder/decoder input/output logging.
- Preserve robot config metadata for exporter startup.

`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp`

- Publishes debug fields over ZMQ:
  - `encoder_obs`
  - `decoder_obs`
  - `decoder_action_raw`
  - `q_target_cmd`
- Republishes `robot_config` periodically.

`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/replay_policy_engine_decoder_test.cpp`

- Standalone C++ TensorRT proof executable.
- Feeds recorded decoder inputs through the same `PolicyEngine` backend used by deploy.

`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/sonic_trt_inference_bridge.cpp`

- Persistent C++ bridge used by Python playback to request TensorRT encoder/decoder inference.

### MuJoCo Substep Debug Capture

`gear_sonic/utils/mujoco_sim/base_sim.py`

- Captures pre-`mj_step` state, controls, q_des, torques, sim time, wall time, ctrl, forces, and `qacc_warmstart`.
- Captures post-`mj_step` qpos/qvel/qacc/actuator force.
- Keeps the latest `0.02 / sim_dt = 4` snapshots for exporter consumption.

## 4. Recorded Datasets On This Branch

### Original SONIC Data

```text
/home/horizon/wrk/SONIC/GR00T-WholeBodyControl/outputs/2026-04-22-19-16-43
```

Purpose:

- Baseline dataset from the original SONIC recording script.
- Useful for comparing normal dataset schema and original behavior.
- Does not contain the full LowCmd / MuJoCo substep debug fields needed for exact substep playback.

### LowCmd Debug Data

```text
/home/horizon/wrk/SONIC/GR00T-WholeBodyControl/outputs/2026-04-29-12-26-28-lowcmd-debug
```

Purpose:

- Records all LowCmd-related command fields and MuJoCo debug state available at each exporter frame.
- Used to investigate action-PD playback, LowCmd timing, and MuJoCo replay fidelity.

Key value:

- This dataset established that command replay needs command fields and MuJoCo state beyond ordinary `action.wbc`.

### ONNX Inputs Debug Data

```text
/home/horizon/wrk/SONIC/GR00T-WholeBodyControl/outputs/onnx-inputs-debug
```

Purpose:

- Records LowCmd debug fields plus exact SONIC encoder/decoder inputs and outputs.
- Main dataset used for Episode 3 playback investigation.
- Enables comparison of recorded teleop inference against Python ONNX playback and C++ TensorRT replay.

Important Episode 3 file:

```text
outputs/onnx-inputs-debug/data/chunk-000/episode_000003.parquet
```

Key findings from this dataset:

- `sonic.encoder_obs` reproduces `token_state` exactly in playback.
- Recorded decoder inputs reproduce decoder outputs exactly when rerun through C++ TensorRT.
- Python ONNXRuntime CPU is close but not bit-identical to TensorRT.
- `robot.motor_pd_substep_q_des` changes inside most 50 Hz frames because MuJoCo reads asynchronous LowCmd at 200 Hz.
- `robot.motor_pd_q_des` equals the latest valid substep q_des in each frame.

## 5. Playback Commands To Keep

Closed-loop ONNX playback:

```bash
MUJOCO_GL=egl ./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_onnx_inputs_pd.py \
  --dataset_dir outputs/onnx-inputs-debug \
  --episode 3 \
  --env_name default \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --fps 50 \
  --free_base \
  --init_substep_frame 1 \
  --init_substep_index 0 \
  --init_substep_all_inputs \
  --start_frame 1 \
  --compare
```

Decoder + physics warmup playback:

```bash
MUJOCO_GL=egl ./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_onnx_inputs_pd_decoder_warmup.py \
  --dataset_dir outputs/onnx-inputs-debug \
  --episode 3 \
  --env_name default \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --fps 50 \
  --free_base \
  --init_substep_frame 1 \
  --init_substep_index 0 \
  --init_substep_all_inputs \
  --start_frame 1 \
  --compare
```

Pure state playback, not substep playback:

This uses the default kinematic path in `playback_lerobot.py`. It directly sets MuJoCo `qpos` from recorded `observation.state` / frame-level state. It does not run ONNX models, does not apply PD physics, and does not restore recorded per-`mj_step` substep inputs.

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot.py \
  --dataset_dir outputs/onnx-inputs-debug \
  --episode 3 \
  --env_name default \
  --fps 50
```

Pure recorded substep playback:

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_action_pd.py \
  --dataset_dir outputs/onnx-inputs-debug \
  --episode 3 \
  --env_name default \
  --fps 50 \
  --free_base \
  --init_substep_frame 1 \
  --init_substep_index 0 \
  --init_substep_all_inputs \
  --start_frame 1 \
  --use_recorded_substep_inputs
```

C++ TensorRT decoder proof:

```bash
cmake --build gear_sonic_deploy/build --target replay_policy_engine_decoder_test -j2

./.venv_sim/bin/python gear_sonic/scripts/compare_recorded_decoder_with_policy_engine.py \
  --dataset_dir outputs/onnx-inputs-debug \
  --episode 3 \
  --model gear_sonic_deploy/policy/release/model_decoder.onnx \
  --executable gear_sonic_deploy/target/release/replay_policy_engine_decoder_test
```

## 6. Next Step

The next planned data collection is:

```text
Start data collection with randomized plate and apple positions on the table.
```

Recommended setup:

- Use the ONNX inputs debug exporter so the next dataset keeps the exact same diagnostic coverage:
  - LowCmd fields
  - MuJoCo substep fields
  - C++ encoder / decoder inputs
  - C++ decoder raw action and final q target
- Add or configure randomization in the table scene before recording:
  - random plate position
  - random apple position
  - keep the random seed / sampled object positions in metadata if possible

