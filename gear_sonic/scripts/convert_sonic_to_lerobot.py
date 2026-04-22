"""Convert SONIC teleoperation recordings (NPZ + JPEG) to LeRobot training format.

Reads episodes saved by record_sonic_teleop.py and writes a LeRobot dataset
(Parquet + H.264 MP4) ready for GR00T N1.5/N1.6 training.

Joint assembly (43 DOF, matching decoupled_wbc modality layout):
  observation.state  = body_q_measured[0:22] ‖ left_hand_q_measured[7]
                       ‖ body_q_measured[22:29] ‖ right_hand_q_measured[7]
  action             = body_q_target[0:22]   ‖ pico.left_hand_joints[7]
                       ‖ body_q_target[22:29]  ‖ pico.right_hand_joints[7]
  observation.eef_state / action.eef  [14]
                     = vr_3pt_pos[0:3] ‖ vr_3pt_ori[0:4]   (L-wrist)
                       ‖ vr_3pt_pos[3:6] ‖ vr_3pt_ori[4:8]  (R-wrist)

The 29-DOF body joint order in SONIC (MuJoCo order) is:
  [0:6]   left_leg  (hip pitch/roll/yaw, knee, ankle pitch/roll)
  [6:12]  right_leg
  [12:15] waist     (yaw, roll, pitch)
  [15:22] left_arm  (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
  [22:29] right_arm

Usage (requires lerobot + av; use the sonic_dc conda env)::

    conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \\
        --input_dir ./recordings \\
        --output_dir ./lerobot_dataset \\
        --task "Pick up apple from table to plate" \\
        --fps 20

    # Single episode:
    conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \\
        --input_dir ./recordings/20260401_115515_ep0002 \\
        --output_dir ./lerobot_dataset \\
        --task "Pick up apple from table to plate"

    # Without images (faster; no MP4 encoding):
    conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \\
        --input_dir ./recordings \\
        --output_dir ./lerobot_dataset \\
        --task "Pick up apple from table to plate" \\
        --no_images

    # Append episodes to an existing dataset (must use same task):
    conda run -n sonic_dc python gear_sonic/scripts/convert_sonic_to_lerobot.py \\
        --input_dir ./recordings_session2 \\
        --output_dir ./lerobot_dataset \\
        --task "Pick up apple from table to plate" \\
        --append
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNKS_SIZE = 1000      # episodes per parquet/video chunk (matches decoupled_wbc)
_CODEBASE_VERSION = "v2.2"
_DEFAULT_BASE_HEIGHT = 0.74  # m  (G1 standing height)
_DEFAULT_NAV_CMD = [0.0, 0.0, 0.0]

# Camera name mapping: SONIC images/ subdir → LeRobot video key
_CAMERA_TO_VIDEO_KEY = {
    "head_camera_left":  "ego_view_left_mono",
    "head_camera_right": "ego_view_right_mono",
    "head_camera":       "ego_view",
}

# ---------------------------------------------------------------------------
# Modality config (matches decoupled_wbc for G1 with hands)
# ---------------------------------------------------------------------------

MODALITY_CONFIG = {
    "state": {
        "left_leg":           {"start": 0,  "end": 6},
        "right_leg":          {"start": 6,  "end": 12},
        "waist":              {"start": 12, "end": 15},
        "left_arm":           {"start": 15, "end": 22},
        "left_hand":          {"start": 22, "end": 29},
        "right_arm":          {"start": 29, "end": 36},
        "right_hand":         {"start": 36, "end": 43},
        "left_wrist_pos":     {"start": 0,  "end": 3,  "original_key": "observation.eef_state"},
        "left_wrist_abs_quat":{"start": 3,  "end": 7,  "original_key": "observation.eef_state",
                               "rotation_type": "quaternion"},
        "right_wrist_pos":    {"start": 7,  "end": 10, "original_key": "observation.eef_state"},
        "right_wrist_abs_quat":{"start":10, "end": 14, "original_key": "observation.eef_state",
                               "rotation_type": "quaternion"},
    },
    "action": {
        "left_leg":           {"start": 0,  "end": 6},
        "right_leg":          {"start": 6,  "end": 12},
        "waist":              {"start": 12, "end": 15},
        "left_arm":           {"start": 15, "end": 22},
        "left_hand":          {"start": 22, "end": 29},
        "right_arm":          {"start": 29, "end": 36},
        "right_hand":         {"start": 36, "end": 43},
        "left_wrist_pos":     {"start": 0,  "end": 3,  "original_key": "action.eef"},
        "left_wrist_abs_quat":{"start": 3,  "end": 7,  "original_key": "action.eef",
                               "rotation_type": "quaternion"},
        "right_wrist_pos":    {"start": 7,  "end": 10, "original_key": "action.eef"},
        "right_wrist_abs_quat":{"start":10, "end": 14, "original_key": "action.eef",
                               "rotation_type": "quaternion"},
        "base_height_command":{"start": 0,  "end": 1,
                               "original_key": "teleop.base_height_command"},
        "navigate_command":   {"start": 0,  "end": 3,
                               "original_key": "teleop.navigate_command"},
    },
    "video": {},       # populated dynamically based on cameras present
    "annotation": {
        "human.task_description": {"original_key": "task_index"},
    },
}

# ---------------------------------------------------------------------------
# Pyarrow schema for the Parquet files
# ---------------------------------------------------------------------------

def _build_parquet_schema(video_keys: list[str]) -> pa.Schema:
    fields = [
        pa.field("observation.state",              pa.list_(pa.float64(), 43)),
        pa.field("observation.eef_state",          pa.list_(pa.float64(), 14)),
        pa.field("action",                         pa.list_(pa.float64(), 43)),
        pa.field("action.eef",                     pa.list_(pa.float64(), 14)),
        pa.field("observation.img_state_delta",    pa.float32()),
        # Active (combined) navigate command — source is mode-dependent (pelvis in POSE, joystick in PLANNER)
        pa.field("teleop.navigate_command",        pa.list_(pa.float64(), 3)),
        # Per-source navigate commands (joystick always present; pelvis is NaN when no foot trackers)
        pa.field("teleop.navigate_cmd_joystick",   pa.list_(pa.float64(), 3)),
        pa.field("teleop.navigate_cmd_pelvis",     pa.list_(pa.float64(), 3)),
        # Base height commands (joystick = button-driven; pelvis = SMPL pelvis Z, NaN when unavailable)
        pa.field("teleop.base_height_cmd_joystick", pa.float64()),
        pa.field("teleop.base_height_cmd_pelvis",   pa.float64()),
        pa.field("robot.base_pos",                 pa.list_(pa.float64(), 3)),
        pa.field("robot.base_quat",                pa.list_(pa.float64(), 4)),
        # Measured robot state history terms used by SONIC decoder
        pa.field("robot.base_ang_vel",             pa.list_(pa.float64(), 3)),
        pa.field("robot.body_dq",                  pa.list_(pa.float64(), 29)),
        # Optional exact model I/O buffers from deploy (when enabled during recording)
        pa.field("sonic.encoder_obs",              pa.list_(pa.float64(), 1762)),
        pa.field("sonic.token_state",              pa.list_(pa.float64(), 64)),
        pa.field("sonic.decoder_obs",              pa.list_(pa.float64(), 994)),
        pa.field("sonic.decoder_action_raw",       pa.list_(pa.float64(), 29)),
        pa.field("sonic.q_target_cmd",             pa.list_(pa.float64(), 29)),
        # SMPL body data (most-recent buffered frame per tick; zeros when SMPL not active)
        # pico.smpl_joints: 24 joints × 3 (absolute world-space XYZ, J=0 is pelvis)
        pa.field("pico.smpl_joints",               pa.list_(pa.float32(), 72)),
        # pico.smpl_pose:   21 joints × 3 (axis-angle per joint)
        pa.field("pico.smpl_pose",                 pa.list_(pa.float32(), 63)),
        # pico.body_root_quat: body root quaternion wxyz from SMPL fit
        pa.field("pico.body_root_quat",            pa.list_(pa.float32(), 4)),
        pa.field("timestamp",                      pa.float32()),
        pa.field("frame_index",                    pa.int64()),
        pa.field("episode_index",                  pa.int64()),
        pa.field("index",                          pa.int64()),
        pa.field("task_index",                     pa.int64()),
    ]
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

def _is_episode_dir(path: Path) -> bool:
    return (path / "pico.npz").is_file() and (path / "sonic.npz").is_file()


def _find_episodes(input_dir: Path) -> list[Path]:
    """Return sorted list of valid episode directories under input_dir."""
    if _is_episode_dir(input_dir):
        return [input_dir]
    dirs = sorted(p for p in input_dir.iterdir() if p.is_dir() and _is_episode_dir(p))
    if not dirs:
        sys.exit(f"[ERROR] No valid episodes found under {input_dir}")
    return dirs


# ---------------------------------------------------------------------------
# Resampling: nearest-neighbour from variable PICO rate to target fps
# ---------------------------------------------------------------------------

def _resample_indices(timestamps: np.ndarray, fps: float) -> np.ndarray:
    """Return source-frame indices for uniform target_fps grid covering the episode.

    timestamps: shape [T] of wall-clock times in seconds (monotonically increasing).
    Returns indices into timestamps, one per target frame, length >= 1.
    """
    t0, t1 = timestamps[0], timestamps[-1]
    duration = t1 - t0
    if duration <= 0:
        return np.array([0], dtype=int)
    n_target = max(1, int(round(duration * fps)))
    target_ts = np.linspace(t0, t1, n_target)
    # nearest-neighbour: for each target timestamp find closest source index
    idx = np.searchsorted(timestamps, target_ts, side="left")
    idx = np.clip(idx, 0, len(timestamps) - 1)
    # pick nearest (compare with previous index)
    prev_idx = np.clip(idx - 1, 0, len(timestamps) - 1)
    closer_prev = np.abs(timestamps[prev_idx] - target_ts) < np.abs(timestamps[idx] - target_ts)
    idx[closer_prev] = prev_idx[closer_prev]
    return idx


# ---------------------------------------------------------------------------
# Joint assembly helpers
# ---------------------------------------------------------------------------

def _assemble_state(body_q: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """Build 43-DOF state vector from 29-DOF body + 7+7 hand joints.

    SONIC body order: [left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
    Target order:     [left_leg(6), right_leg(6), waist(3), left_arm(7), left_hand(7),
                       right_arm(7), right_hand(7)]
    """
    return np.concatenate([
        body_q[0:22],     # left_leg + right_leg + waist + left_arm
        left_hand,        # left_hand (7)
        body_q[22:29],    # right_arm
        right_hand,       # right_hand (7)
    ]).astype(np.float64)


def _assemble_eef(vr_pos: np.ndarray, vr_ori: np.ndarray) -> np.ndarray:
    """Build 14-DOF EEF state: [L-pos(3), L-quat(4), R-pos(3), R-quat(4)]."""
    return np.concatenate([
        vr_pos[0:3],   # L-wrist xyz
        vr_ori[0:4],   # L-wrist wxyz
        vr_pos[3:6],   # R-wrist xyz
        vr_ori[4:8],   # R-wrist wxyz
    ]).astype(np.float64)


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------

class _VideoEncoder:
    """Lightweight wrapper around PyAV for H.264 MP4 output."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("h264", rate=int(fps))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"x264-params": "log-level=0"}

    def add_frame(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():  # flush
            self._container.mux(packet)
        self._container.close()


def _load_jpeg(path: Path) -> np.ndarray | None:
    """Decode a JPEG file to an RGB numpy array [H, W, 3]."""
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                return frame.to_ndarray(format="rgb24")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _compute_stats(arrays: dict[str, list[np.ndarray]]) -> dict[str, dict]:
    stats = {}
    for key, rows in arrays.items():
        mat = np.stack(rows, axis=0).astype(np.float64)
        stats[key] = {
            "min":   mat.min(axis=0).tolist(),
            "max":   mat.max(axis=0).tolist(),
            "mean":  mat.mean(axis=0).tolist(),
            "std":   mat.std(axis=0).tolist(),
            "count": [len(mat)] * mat.shape[-1] if mat.ndim > 1 else [len(mat)],
        }
    return stats


# ---------------------------------------------------------------------------
# Single-episode conversion
# ---------------------------------------------------------------------------

def _read_episode_meta(ep_dir: Path) -> dict:
    """Load meta.json from an episode directory (returns {} if absent)."""
    meta_path = ep_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _convert_episode(
    ep_dir: Path,
    out_dir: Path,
    episode_index: int,
    global_frame_offset: int,
    task_index: int,
    fps: float,
    include_images: bool,
    video_keys: list[str],
    schema: pa.Schema,
) -> tuple[int, dict]:  # (n_frames, stats_dict)

    pico = np.load(ep_dir / "pico.npz")
    sonic = np.load(ep_dir / "sonic.npz")

    # ------------------------------------------------------------------
    # Timestamps and resampling
    # ------------------------------------------------------------------
    ts = pico["timestamp_realtime"].reshape(-1)
    src_idx = _resample_indices(ts, fps)
    n_frames = len(src_idx)
    t0 = ts[src_idx[0]]

    # ------------------------------------------------------------------
    # Field extraction helpers (with backward-compat fallbacks)
    # ------------------------------------------------------------------
    def pico_f(key: str, fallback: np.ndarray) -> np.ndarray:
        return pico[key] if key in pico else np.tile(fallback, (len(ts), 1)).reshape(len(ts), -1)

    def sonic_f(key: str, fallback: np.ndarray) -> np.ndarray:
        return sonic[key] if key in sonic else np.tile(fallback, (len(sonic["body_q_measured"]), 1)).reshape(len(sonic["body_q_measured"]), -1)

    body_q_meas   = sonic["body_q_measured"]        # [T, 29]
    body_q_tgt    = sonic["body_q_target"]          # [T, 29]
    lh_meas       = sonic["left_hand_q_measured"]   # [T, 7]
    rh_meas       = sonic["right_hand_q_measured"]  # [T, 7]
    vr_pos        = sonic["vr_3point_position"]     # [T, 9]
    vr_ori        = sonic["vr_3point_orientation"]  # [T, 12]
    base_pos      = sonic_f("base_pos_sim",  np.zeros(3, dtype=np.float64))   # [T, 3] XYZ world frame
    base_quat     = sonic_f("base_quat_sim", np.array([1., 0., 0., 0.], dtype=np.float64))  # [T, 4] wxyz
    # Decoder-history state from g1_debug (fallback to zeros for older recordings)
    base_ang_vel_meas = sonic_f("base_ang_vel_measured", np.zeros(3, dtype=np.float64))  # [T, 3]
    body_dq_meas      = sonic_f("body_dq_measured", np.zeros(29, dtype=np.float64))       # [T, 29]
    # Optional exact model I/O (when --enable-model-io-recording is used in deploy)
    encoder_obs       = sonic_f("encoder_obs", np.zeros(1762, dtype=np.float64))           # [T, 1762]
    token_state       = sonic_f("token_state", np.zeros(64, dtype=np.float64))             # [T, 64]
    decoder_obs       = sonic_f("decoder_obs", np.zeros(994, dtype=np.float64))            # [T, 994]
    decoder_action    = sonic_f("decoder_action_raw", np.zeros(29, dtype=np.float64))      # [T, 29]
    q_target_cmd      = sonic_f("q_target_cmd", np.zeros(29, dtype=np.float64))            # [T, 29]

    # pico hand targets — shape [T, 7]
    lh_tgt = pico["left_hand_joints"]   # always present
    rh_tgt = pico["right_hand_joints"]

    # navigate_cmd — shape [T, 3]; zeros if not recorded
    nav_cmd = pico_f("navigate_cmd", np.array(_DEFAULT_NAV_CMD, dtype=np.float32))

    # Per-source navigate commands (present in newer recordings only)
    _zeros3 = np.zeros(3, dtype=np.float32)
    _nans3  = np.full(3, np.nan, dtype=np.float32)
    nav_cmd_joystick = pico_f("navigate_cmd_joystick", _zeros3)
    nav_cmd_pelvis   = pico_f("navigate_cmd_pelvis",   _nans3)

    # Base height — joystick source; shape [T]; default 0.74 if not recorded
    if "base_height_cmd_joystick" in pico:
        bh_joystick = pico["base_height_cmd_joystick"].reshape(-1)
    elif "base_height_command" in pico:
        bh_joystick = pico["base_height_command"].reshape(-1)
    else:
        bh_joystick = np.full(len(ts), _DEFAULT_BASE_HEIGHT, dtype=np.float32)

    # Base height — pelvis source; NaN when foot trackers unavailable or field absent
    if "base_height_cmd_pelvis" in pico:
        bh_pelvis = pico["base_height_cmd_pelvis"].reshape(-1)
    else:
        bh_pelvis = np.full(len(ts), np.nan, dtype=np.float32)

    # SMPL body data — most-recent buffered frame per tick (index -1 along the N axis)
    # shape [T, N, 24, 3] → take last frame → [T, 72]
    if "smpl_joints" in pico:
        smpl_joints = pico["smpl_joints"][:, -1, :, :].reshape(len(ts), 72)
    else:
        smpl_joints = np.zeros((len(ts), 72), dtype=np.float32)

    if "smpl_pose" in pico:
        smpl_pose = pico["smpl_pose"][:, -1, :, :].reshape(len(ts), 63)
    else:
        smpl_pose = np.zeros((len(ts), 63), dtype=np.float32)

    # body_quat_w — shape [T, N, 4] → most-recent frame → [T, 4]
    if "body_quat_w" in pico:
        body_root_quat = pico["body_quat_w"][:, -1, :].astype(np.float32)
    else:
        body_root_quat = np.tile([1.0, 0.0, 0.0, 0.0], (len(ts), 1)).astype(np.float32)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    encoders: dict[str, _VideoEncoder] = {}
    if include_images and video_keys:
        chunk = episode_index // _CHUNKS_SIZE
        for vk in video_keys:
            mp4_path = out_dir / "videos" / f"chunk-{chunk:03d}" / vk / f"episode_{episode_index:06d}.mp4"
            # Probe first JPEG for dimensions
            cam_name = next(c for c, k in _CAMERA_TO_VIDEO_KEY.items() if k == vk)
            sample_jpg = ep_dir / "images" / cam_name / "000000.jpg"
            sample_rgb = _load_jpeg(sample_jpg)
            if sample_rgb is None:
                print(f"  [WARN] Could not read {sample_jpg}; skipping video key {vk}")
                continue
            h, w = sample_rgb.shape[:2]
            encoders[vk] = _VideoEncoder(mp4_path, w, h, fps)

    # ------------------------------------------------------------------
    # Build per-frame data
    # ------------------------------------------------------------------
    rows: dict[str, list] = {f: [] for f in schema.names}
    stat_accum: dict[str, list[np.ndarray]] = {
        k: [] for k in ["observation.state", "observation.eef_state",
                         "action", "action.eef",
                         "teleop.navigate_command",
                         "teleop.navigate_cmd_joystick", "teleop.navigate_cmd_pelvis",
                         "teleop.base_height_cmd_joystick", "teleop.base_height_cmd_pelvis",
                         "robot.base_pos", "robot.base_quat", "robot.base_ang_vel", "robot.body_dq",
                         "sonic.encoder_obs", "sonic.token_state", "sonic.decoder_obs",
                         "sonic.decoder_action_raw", "sonic.q_target_cmd",
                         "pico.smpl_joints", "pico.body_root_quat"]
    }

    for out_fi, si in enumerate(src_idx):
        state = _assemble_state(body_q_meas[si], lh_meas[si], rh_meas[si])
        action = _assemble_state(body_q_tgt[si], lh_tgt[si].astype(np.float64), rh_tgt[si].astype(np.float64))
        eef = _assemble_eef(vr_pos[si], vr_ori[si])
        nav = nav_cmd[si].astype(np.float64)
        nav_js = nav_cmd_joystick[si].astype(np.float64)
        nav_pl = nav_cmd_pelvis[si].astype(np.float64)
        bh_js = float(bh_joystick[si])
        bh_pl = float(bh_pelvis[si])
        timestamp = float(ts[si] - t0)

        bp = base_pos[si].astype(np.float64)
        bq = base_quat[si].astype(np.float64)
        bav = base_ang_vel_meas[si].astype(np.float64)
        bdq = body_dq_meas[si].astype(np.float64)
        eobs = encoder_obs[si].astype(np.float64)
        tks = token_state[si].astype(np.float64)
        dobs = decoder_obs[si].astype(np.float64)
        draw = decoder_action[si].astype(np.float64)
        qcmd = q_target_cmd[si].astype(np.float64)

        sj  = smpl_joints[si].astype(np.float32)
        sp  = smpl_pose[si].astype(np.float32)
        brq = body_root_quat[si].astype(np.float32)

        rows["observation.state"].append(state.tolist())
        rows["observation.eef_state"].append(eef.tolist())
        rows["action"].append(action.tolist())
        rows["action.eef"].append(eef.tolist())
        rows["observation.img_state_delta"].append(np.float32(0.0))
        rows["teleop.navigate_command"].append(nav.tolist())
        rows["teleop.navigate_cmd_joystick"].append(nav_js.tolist())
        rows["teleop.navigate_cmd_pelvis"].append(nav_pl.tolist())
        rows["teleop.base_height_cmd_joystick"].append(bh_js)
        rows["teleop.base_height_cmd_pelvis"].append(bh_pl)
        rows["robot.base_pos"].append(bp.tolist())
        rows["robot.base_quat"].append(bq.tolist())
        rows["robot.base_ang_vel"].append(bav.tolist())
        rows["robot.body_dq"].append(bdq.tolist())
        rows["sonic.encoder_obs"].append(eobs.tolist())
        rows["sonic.token_state"].append(tks.tolist())
        rows["sonic.decoder_obs"].append(dobs.tolist())
        rows["sonic.decoder_action_raw"].append(draw.tolist())
        rows["sonic.q_target_cmd"].append(qcmd.tolist())
        rows["pico.smpl_joints"].append(sj.tolist())
        rows["pico.smpl_pose"].append(sp.tolist())
        rows["pico.body_root_quat"].append(brq.tolist())
        rows["timestamp"].append(np.float32(timestamp))
        rows["frame_index"].append(out_fi)
        rows["episode_index"].append(episode_index)
        rows["index"].append(global_frame_offset + out_fi)
        rows["task_index"].append(task_index)

        stat_accum["observation.state"].append(state)
        stat_accum["observation.eef_state"].append(eef)
        stat_accum["action"].append(action)
        stat_accum["action.eef"].append(eef)
        stat_accum["teleop.navigate_command"].append(nav)
        stat_accum["teleop.navigate_cmd_joystick"].append(nav_js)
        stat_accum["teleop.navigate_cmd_pelvis"].append(nav_pl)
        stat_accum["teleop.base_height_cmd_joystick"].append(np.array([bh_js]))
        stat_accum["teleop.base_height_cmd_pelvis"].append(np.array([bh_pl]))
        stat_accum["robot.base_pos"].append(bp)
        stat_accum["robot.base_quat"].append(bq)
        stat_accum["robot.base_ang_vel"].append(bav)
        stat_accum["robot.body_dq"].append(bdq)
        stat_accum["sonic.encoder_obs"].append(eobs)
        stat_accum["sonic.token_state"].append(tks)
        stat_accum["sonic.decoder_obs"].append(dobs)
        stat_accum["sonic.decoder_action_raw"].append(draw)
        stat_accum["sonic.q_target_cmd"].append(qcmd)
        stat_accum["pico.smpl_joints"].append(sj)
        stat_accum["pico.body_root_quat"].append(brq)

        # Images
        if encoders:
            # JPEG filenames are named by source pose-tick index
            for vk, enc in encoders.items():
                cam_name = next(c for c, k in _CAMERA_TO_VIDEO_KEY.items() if k == vk)
                jpg = ep_dir / "images" / cam_name / f"{si:06d}.jpg"
                rgb = _load_jpeg(jpg)
                if rgb is not None:
                    enc.add_frame(rgb)
                else:
                    # Fallback: black frame
                    h, w = enc._stream.height, enc._stream.width
                    enc.add_frame(np.zeros((h, w, 3), dtype=np.uint8))

    # ------------------------------------------------------------------
    # Write Parquet
    # ------------------------------------------------------------------
    chunk = episode_index // _CHUNKS_SIZE
    pq_path = out_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    pq_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {name: pa.array(rows[name], type=schema.field(name).type) for name in schema.names}
    table = pa.table(arrays, schema=schema)
    pq.write_table(table, str(pq_path))

    # Close video encoders
    for enc in encoders.values():
        enc.close()

    stats = _compute_stats(stat_accum)
    return n_frames, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SONIC NPZ recordings to LeRobot training format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir", required=True, type=Path,
                   help="Directory containing episode subdirectories (or a single episode dir).")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Root directory for the LeRobot dataset (created if absent).")
    p.add_argument("--task", default=None,
                   help="Language task description. If omitted, read from each episode's meta.json "
                        "(set by record_sonic_teleop.py --task). Required if meta.json has no task.")
    p.add_argument("--fps", type=float, default=20.0,
                   help="Target frame rate for the output dataset (default: 20 Hz).")
    p.add_argument("--no_images", action="store_true",
                   help="Skip video encoding even if images/ directories are present.")
    p.add_argument("--append", action="store_true",
                   help="Append episodes to an existing dataset at output_dir.")
    p.add_argument("--robot_type", default="g1",
                   help="Robot type string written to info.json (default: g1).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    episodes = _find_episodes(args.input_dir)
    print(f"Found {len(episodes)} episode(s) in {args.input_dir}")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Detect which cameras / video keys are present across episodes
    # ------------------------------------------------------------------
    video_keys: list[str] = []
    if not args.no_images:
        for ep_dir in episodes:
            img_dir = ep_dir / "images"
            if img_dir.is_dir():
                cams = sorted(d.name for d in img_dir.iterdir() if d.is_dir())
                video_keys = [_CAMERA_TO_VIDEO_KEY[c] for c in cams if c in _CAMERA_TO_VIDEO_KEY]
                break  # use first episode with images as reference

    modality_config = dict(MODALITY_CONFIG)
    modality_config["video"] = {vk: {"original_key": f"observation.images.{vk}"}
                                for vk in video_keys}

    schema = _build_parquet_schema(video_keys)

    # ------------------------------------------------------------------
    # Load or initialise dataset state (for --append)
    # ------------------------------------------------------------------
    info_path = meta_dir / "info.json"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    ep_stats_path = meta_dir / "episodes_stats.jsonl"
    modality_path = meta_dir / "modality.json"

    # task_registry maps task_str -> task_index; populated from tasks.jsonl when appending
    task_registry: dict[str, int] = {}

    if args.append and info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        with open(tasks_path) as f:
            for line in f:
                rec = json.loads(line)
                task_registry[rec["task"]] = rec["task_index"]
        start_episode = info["total_episodes"]
        start_frame = info["total_frames"]
    else:
        # Fresh dataset — tasks.jsonl written per-episode below
        start_episode = 0
        start_frame = 0
        tasks_path.write_text("")  # empty file; entries appended per episode

    def _resolve_task_index(task_str: str) -> int:
        """Return existing task_index or register a new one."""
        if task_str in task_registry:
            return task_registry[task_str]
        new_idx = len(task_registry)
        task_registry[task_str] = new_idx
        with open(tasks_path, "a") as f:
            f.write(json.dumps({"task_index": new_idx, "task": task_str}) + "\n")
        return new_idx

    # Build features dict for info.json (only for fresh datasets)
    if not (args.append and info_path.exists()):
        features: dict[str, Any] = {
            "observation.state":         {"dtype": "float64", "shape": [43], "names": None},
            "observation.eef_state":     {"dtype": "float64", "shape": [14], "names": None},
            "action":                    {"dtype": "float64", "shape": [43], "names": None},
            "action.eef":                {"dtype": "float64", "shape": [14], "names": None},
            "observation.img_state_delta": {"dtype": "float32", "shape": [1], "names": None},
            "teleop.navigate_command":         {"dtype": "float64", "shape": [3],
                                               "names": ["lin_vel_x", "lin_vel_y", "ang_vel_z"]},
            "teleop.navigate_cmd_joystick":    {"dtype": "float64", "shape": [3],
                                               "names": ["lin_vel_x", "lin_vel_y", "ang_vel_z"]},
            "teleop.navigate_cmd_pelvis":      {"dtype": "float64", "shape": [3],
                                               "names": ["lin_vel_x", "lin_vel_y", "ang_vel_z"]},
            "teleop.base_height_cmd_joystick": {"dtype": "float64", "shape": [1],
                                               "names": ["base_height_command"]},
            "teleop.base_height_cmd_pelvis":   {"dtype": "float64", "shape": [1],
                                               "names": ["base_height_command"]},
            "pico.smpl_joints":                {"dtype": "float32", "shape": [72],
                                               "names": None},
            "pico.smpl_pose":                  {"dtype": "float32", "shape": [63],
                                               "names": None},
            "pico.body_root_quat":             {"dtype": "float32", "shape": [4],
                                               "names": ["w", "x", "y", "z"]},
            "robot.base_pos":            {"dtype": "float64", "shape": [3],
                                          "names": ["x", "y", "z"]},
            "robot.base_quat":           {"dtype": "float64", "shape": [4],
                                          "names": ["w", "x", "y", "z"]},
            "robot.base_ang_vel":        {"dtype": "float64", "shape": [3],
                                          "names": ["wx", "wy", "wz"]},
            "robot.body_dq":             {"dtype": "float64", "shape": [29], "names": None},
            "sonic.encoder_obs":         {"dtype": "float64", "shape": [1762], "names": None},
            "sonic.token_state":         {"dtype": "float64", "shape": [64], "names": None},
            "sonic.decoder_obs":         {"dtype": "float64", "shape": [994], "names": None},
            "sonic.decoder_action_raw":  {"dtype": "float64", "shape": [29], "names": None},
            "sonic.q_target_cmd":        {"dtype": "float64", "shape": [29], "names": None},
            "timestamp":   {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index":{"dtype":"int64",   "shape": [1], "names": None},
            "index":       {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":  {"dtype": "int64",   "shape": [1], "names": None},
        }
        for vk in video_keys:
            features[f"observation.images.{vk}"] = {
                "dtype": "video", "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
            }

        info = {
            "codebase_version": _CODEBASE_VERSION,
            "robot_type": args.robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 0,
            "chunks_size": _CHUNKS_SIZE,
            "fps": args.fps,
            "splits": {"train": "0:0"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "discarded_episode_indices": [],
            "data_collection_info": {
                "lower_body_policy": "sonic",
                "wbc_model_path": "",
                "teleoperator_username": "",
                "robot_type": args.robot_type,
                "robot_id": "sim",
            },
            "features": features,
        }

    # Write modality config (always overwrite to keep in sync)
    with open(modality_path, "w") as f:
        json.dump(modality_config, f, indent=4)

    # ------------------------------------------------------------------
    # Convert each episode
    # ------------------------------------------------------------------
    total_new_frames = 0
    ep_open_mode = "a" if args.append else "w"

    with open(episodes_path, ep_open_mode) as ep_f, \
         open(ep_stats_path, ep_open_mode) as st_f:

        for local_i, ep_dir in enumerate(episodes):
            ep_idx = start_episode + local_i

            # Resolve task string: CLI flag takes precedence; fall back to meta.json
            ep_meta = _read_episode_meta(ep_dir)
            task_str = args.task or ep_meta.get("task") or ""
            if not task_str:
                sys.exit(
                    f"[ERROR] No task description for {ep_dir.name}. "
                    "Pass --task or re-record with record_sonic_teleop.py --task <description>."
                )
            ep_task_index = _resolve_task_index(task_str)
            ep_env_name = ep_meta.get("env_name", "")

            print(f"  [{local_i+1}/{len(episodes)}] {ep_dir.name}  →  episode_{ep_idx:06d}"
                  f"  task='{task_str}'"
                  + (f"  env={ep_env_name}" if ep_env_name else ""))

            n_frames, stats = _convert_episode(
                ep_dir=ep_dir,
                out_dir=out_dir,
                episode_index=ep_idx,
                global_frame_offset=start_frame + total_new_frames,
                task_index=ep_task_index,
                fps=args.fps,
                include_images=(not args.no_images),
                video_keys=video_keys,
                schema=schema,
            )

            total_new_frames += n_frames
            ep_f.write(json.dumps({
                "episode_index": ep_idx,
                "tasks": [task_str],
                "length": n_frames,
            }) + "\n")
            st_f.write(json.dumps({
                "episode_index": ep_idx,
                "stats": stats,
            }) + "\n")

            print(f"    {n_frames} frames @ {args.fps} Hz")

    # ------------------------------------------------------------------
    # Update info.json
    # ------------------------------------------------------------------
    new_total_ep = start_episode + len(episodes)
    new_total_fr = start_frame + total_new_frames
    n_video_keys = len(video_keys) if not args.no_images else 0
    info["total_episodes"] = new_total_ep
    info["total_frames"] = new_total_fr
    info["total_videos"] = new_total_ep * n_video_keys
    info["total_chunks"] = max(1, (new_total_ep + _CHUNKS_SIZE - 1) // _CHUNKS_SIZE)
    info["splits"] = {"train": f"0:{new_total_ep}"}

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nDone. Dataset at {out_dir}")
    print(f"  Episodes: {new_total_ep}  |  Frames: {new_total_fr}  |  Videos: {info['total_videos']}")


if __name__ == "__main__":
    main()
