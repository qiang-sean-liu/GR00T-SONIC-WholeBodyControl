"""Replay recorded SONIC/WBC actions through MuJoCo PD control.

This is action-based playback: it loads ``action`` / ``action.wbc`` from a
LeRobot episode and reconstructs joint-position targets. The physics
step matches ``gear_sonic/utils/mujoco_sim/base_sim.py`` (teleop sim): PD
torques ``tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)`` on the body,
with the same optional floating-base drive as the SONIC ``playback_lerobot``
physics path.

**Per-joint command fields (29-DOF body), when present in the Parquet:**

- ``sonic.q_target_cmd`` — overrides body position targets. If missing, body
  targets come from ``action.wbc``. Use ``--body_action_format raw_policy`` for
  deploy post-processing:
  ``default_angles + raw_action[isaaclab_to_mujoco] * g1_action_scale``.
- ``robot.motor_kp``, ``robot.motor_kd`` (aliases: ``robot.body_kp``,
  ``robot.body_kd``)
- ``robot.motor_tau`` (alias: ``robot.body_tau``) — feedforward ``tau`` on the
  command, matching ``low_cmd.motor_cmd[i].tau``.
- ``robot.motor_tau_est`` — measured/estimated real-time motor torque. Use
  ``--use_recorded_torque`` to bypass PD and replay this torque directly.
- ``robot.motor_dq`` (alias: ``robot.body_dq_cmd``) — command velocity
  ``dq_des``, matching ``low_cmd.motor_cmd[i].dq``.

Missing columns are filled from the WBC YAML (``MOTOR_KP`` / ``MOTOR_KD``), or
if the YAML is unavailable, from the C++-style armature defaults in
``playback_lerobot`` (``_KP_BODY29`` / ``_KD_BODY29``). ``tau_ff`` and
``dq_des`` default to zero. ``robot.body_dq`` (measured state) is **not** used
as a command; use ``robot.motor_dq`` / ``robot.body_dq_cmd`` for that.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq
import yaml

from gear_sonic.scripts.playback_lerobot import (
    _ACTION_SCALE,
    _BODY_IDX,
    _DEFAULT_ANGLES,
    _ISAACLAB_TO_MUJOCO,
    _KD_BODY29,
    _KP_BODY29,
    _LEFT_HAND_IDX,
    _RIGHT_HAND_IDX,
    _ROOT_HEIGHT,
    _SIM_DT,
    _SIM_STEPS_PER_POLICY,
    _TORQUE_LIMIT_BODY29,
    _TORQUE_LIMIT_HAND,
    _build_ctrl_map,
    _build_joint_indices,
    _load_xml,
    _quat_conj,
    _quat_mult,
    _quat_slerp,
    _quat_to_rot_matrix,
    _set_qpos,
)
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WAIST_SLICE_BODY29 = slice(12, 15)  # waist_yaw, waist_roll, waist_pitch
_SUBSTEP_TIME_EPS = 1e-12
_HAND_KP_DEFAULT = 1.5
_HAND_KD_DEFAULT = 0.1
_HAND_MAX_DELTA_Q = 0.25
_HAND_MAX_CLOSE_RATIO = 1.0
_HAND_MAX_LIMITS_LEFT = np.array([1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
_HAND_MIN_LIMITS_LEFT = np.array([-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75], dtype=np.float64)
_HAND_MAX_LIMITS_RIGHT = np.array([1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75], dtype=np.float64)
_HAND_MIN_LIMITS_RIGHT = np.array([-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
_BODY_ACTION_FORMATS = ("auto", "q_target", "raw_policy")
_ELASTIC_BAND_BODIES = ("auto", "pelvis", "torso_link")


def _clip_hand_to_max_open(
    desired_q: float, max_limit: float, min_limit: float, max_close_ratio: float
) -> float:
    """Mirror Dex3Hands::clipToMaxOpen for one hand motor."""
    q_max_open_pos = max_close_ratio * max_limit
    q_max_open_neg = max_close_ratio * min_limit
    if desired_q > 0.0 and max_limit > 0.0 and desired_q > q_max_open_pos:
        return float(q_max_open_pos)
    if desired_q < 0.0 and min_limit < 0.0 and desired_q < q_max_open_neg:
        return float(q_max_open_neg)
    return float(desired_q)


def _dex3_smoothed_hand_target(
    target_arr: np.ndarray,
    current_q: np.ndarray,
    max_limits: np.ndarray,
    min_limits: np.ndarray,
    max_close_ratio: float = _HAND_MAX_CLOSE_RATIO,
) -> np.ndarray:
    """Mirror Dex3Hands::writeOnce target clipping and delta-q smoothing."""
    out = np.zeros_like(target_arr, dtype=np.float64)
    for i, target_q in enumerate(target_arr):
        clipped = _clip_hand_to_max_open(
            float(target_q), max_limits[i], min_limits[i], max_close_ratio
        )
        delta = clipped - current_q[i]
        out[i] = current_q[i] + np.clip(delta, -_HAND_MAX_DELTA_Q, _HAND_MAX_DELTA_Q)
    return out


def _wbc_yaml_default() -> Path:
    return _REPO_ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"


def _load_motor_gains_wbc_yaml(
    path: str | None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Load 29-DOF ``MOTOR_KP`` / ``MOTOR_KD`` from WBC yaml. Returns (kp, kd, err)."""
    p = Path(path) if path else _wbc_yaml_default()
    if not p.is_file():
        return _KP_BODY29.copy(), _KD_BODY29.copy(), f"(missing: {p})"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "MOTOR_KP" not in cfg or "MOTOR_KD" not in cfg:
        return _KP_BODY29.copy(), _KD_BODY29.copy(), f"(invalid: {p})"
    kp = np.asarray(cfg["MOTOR_KP"], dtype=np.float64).reshape(29)
    kd = np.asarray(cfg["MOTOR_KD"], dtype=np.float64).reshape(29)
    if kp.size != 29 or kd.size != 29:
        return _KP_BODY29.copy(), _KD_BODY29.copy(), f"(bad len in {p})"
    return kp, kd, None


def _episode_path(dataset_dir: str, episode: int) -> str:
    chunk = episode // 1000
    return os.path.join(
        dataset_dir,
        f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
    )


def _load_optional_29(
    table,
    col_primary: str,
    col_alias: str | None,
) -> np.ndarray | None:
    names = table.schema.names
    if col_primary in names:
        c = col_primary
    elif col_alias and col_alias in names:
        c = col_alias
    else:
        return None
    arr = np.array([r.as_py() for r in table.column(c)], dtype=np.float64)
    T = table.num_rows
    if arr.shape != (T, 29):
        raise ValueError(
            f"Column {c!r} must have shape (T, 29); got {arr.shape} for T={T}"
        )
    return arr


def _load_optional_n(
    table,
    n: int,
    col_primary: str,
    *aliases: str,
) -> np.ndarray | None:
    names = table.schema.names
    c = next((name for name in (col_primary, *aliases) if name in names), None)
    if c is None:
        return None
    arr = np.array([r.as_py() for r in table.column(c)], dtype=np.float64)
    T = table.num_rows
    if arr.shape != (T, n):
        raise ValueError(
            f"Column {c!r} must have shape (T, {n}); got {arr.shape} for T={T}"
        )
    return arr


def _load_optional_vector(table, col_primary: str, *aliases: str) -> np.ndarray | None:
    names = table.schema.names
    c = next((name for name in (col_primary, *aliases) if name in names), None)
    if c is None:
        return None
    arr = np.array([r.as_py() for r in table.column(c)], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != table.num_rows:
        raise ValueError(
            f"Column {c!r} must have shape (T, N); got {arr.shape} for T={table.num_rows}"
        )
    return arr


def _load_action_episode(dataset_dir: str, episode: int) -> dict:
    path = _episode_path(dataset_dir, episode)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")

    schema = pq.read_schema(path)
    action_col = "action.wbc" if "action.wbc" in schema.names else "action"
    if action_col not in schema.names:
        raise ValueError("Expected an action column named 'action.wbc' or 'action'.")

    cols: list[str] = ["observation.state", action_col, "task_index"]
    has_base_pos = "robot.base_pos" in schema.names
    base_quat_col = None
    if "robot.base_quat" in schema.names:
        base_quat_col = "robot.base_quat"
    elif "observation.root_orientation" in schema.names:
        base_quat_col = "observation.root_orientation"

    if has_base_pos:
        cols.append("robot.base_pos")
    if base_quat_col is not None:
        cols.append(base_quat_col)

    for opt in (
        "sonic.q_target_cmd",
        "robot.motor_q",
        "robot.motor_kp",
        "robot.body_kp",
        "robot.motor_kd",
        "robot.body_kd",
        "robot.motor_tau",
        "robot.body_tau",
        "robot.motor_tau_est",
        "robot.motor_dq",
        "robot.body_dq_cmd",
        "robot.left_hand_motor_q",
        "robot.left_hand_motor_dq",
        "robot.left_hand_motor_kp",
        "robot.left_hand_motor_kd",
        "robot.left_hand_motor_tau",
        "robot.left_hand_motor_tau_est",
        "robot.right_hand_motor_q",
        "robot.right_hand_motor_dq",
        "robot.right_hand_motor_kp",
        "robot.right_hand_motor_kd",
        "robot.right_hand_motor_tau",
        "robot.right_hand_motor_tau_est",
        "robot.mujoco_qpos",
        "robot.mujoco_qvel",
        "robot.motor_pd_substep_sim_time",
        "robot.mujoco_substep_qpos",
        "robot.mujoco_substep_qvel",
        "robot.mujoco_substep_ctrl",
        "robot.mujoco_substep_qfrc_applied",
        "robot.mujoco_substep_xfrc_applied",
        "robot.mujoco_substep_qacc_warmstart",
    ):
        if opt in schema.names and opt not in cols:
            cols.append(opt)

    table = pq.read_table(path, columns=cols)
    T = table.num_rows
    states = np.array(
        [r.as_py() for r in table.column("observation.state")], dtype=np.float64
    )
    actions = np.array([r.as_py() for r in table.column(action_col)], dtype=np.float64)
    task_indices = table.column("task_index").to_pylist()
    base_pos = (
        np.array([r.as_py() for r in table.column("robot.base_pos")], dtype=np.float64)
        if has_base_pos
        else None
    )
    base_quat = (
        np.array([r.as_py() for r in table.column(base_quat_col)], dtype=np.float64)
        if base_quat_col is not None
        else None
    )

    def load_q29(primary: str) -> np.ndarray | None:
        if primary not in table.schema.names:
            return None
        a = np.array([r.as_py() for r in table.column(primary)], dtype=np.float64)
        if a.shape != (T, 29):
            raise ValueError(
                f"Column {primary!r} must be (T, 29); got {a.shape} T={T}"
            )
        return a

    q_cmd = load_q29("sonic.q_target_cmd")
    motor_q = _load_optional_29(table, "robot.motor_q", None)
    motor_kp = _load_optional_29(table, "robot.motor_kp", "robot.body_kp")
    motor_kd = _load_optional_29(table, "robot.motor_kd", "robot.body_kd")
    motor_tau = _load_optional_29(table, "robot.motor_tau", "robot.body_tau")
    motor_tau_est = _load_optional_29(table, "robot.motor_tau_est", None)
    motor_dq = _load_optional_29(table, "robot.motor_dq", "robot.body_dq_cmd")
    left_hand_motor_q = _load_optional_n(table, 7, "robot.left_hand_motor_q")
    left_hand_motor_dq = _load_optional_n(table, 7, "robot.left_hand_motor_dq")
    left_hand_motor_kp = _load_optional_n(table, 7, "robot.left_hand_motor_kp")
    left_hand_motor_kd = _load_optional_n(table, 7, "robot.left_hand_motor_kd")
    left_hand_motor_tau = _load_optional_n(table, 7, "robot.left_hand_motor_tau")
    left_hand_motor_tau_est = _load_optional_n(table, 7, "robot.left_hand_motor_tau_est")
    right_hand_motor_q = _load_optional_n(table, 7, "robot.right_hand_motor_q")
    right_hand_motor_dq = _load_optional_n(table, 7, "robot.right_hand_motor_dq")
    right_hand_motor_kp = _load_optional_n(table, 7, "robot.right_hand_motor_kp")
    right_hand_motor_kd = _load_optional_n(table, 7, "robot.right_hand_motor_kd")
    right_hand_motor_tau = _load_optional_n(table, 7, "robot.right_hand_motor_tau")
    right_hand_motor_tau_est = _load_optional_n(table, 7, "robot.right_hand_motor_tau_est")
    mujoco_qpos = _load_optional_vector(table, "robot.mujoco_qpos")
    mujoco_qvel = _load_optional_vector(table, "robot.mujoco_qvel")
    substep_time = _load_optional_vector(table, "robot.motor_pd_substep_sim_time")
    substep_qpos = _load_optional_vector(table, "robot.mujoco_substep_qpos")
    substep_qvel = _load_optional_vector(table, "robot.mujoco_substep_qvel")
    substep_ctrl = _load_optional_vector(table, "robot.mujoco_substep_ctrl")
    substep_qfrc = _load_optional_vector(table, "robot.mujoco_substep_qfrc_applied")
    substep_xfrc = _load_optional_vector(table, "robot.mujoco_substep_xfrc_applied")
    substep_warm = _load_optional_vector(table, "robot.mujoco_substep_qacc_warmstart")

    return {
        "states": states,
        "actions": actions,
        "task_indices": task_indices,
        "base_pos": base_pos,
        "base_quat": base_quat,
        "q_target_cmd": q_cmd,
        "motor_q": motor_q,
        "motor_kp": motor_kp,
        "motor_kd": motor_kd,
        "motor_tau": motor_tau,
        "motor_tau_est": motor_tau_est,
        "motor_dq": motor_dq,
        "left_hand_motor_q": left_hand_motor_q,
        "left_hand_motor_dq": left_hand_motor_dq,
        "left_hand_motor_kp": left_hand_motor_kp,
        "left_hand_motor_kd": left_hand_motor_kd,
        "left_hand_motor_tau": left_hand_motor_tau,
        "left_hand_motor_tau_est": left_hand_motor_tau_est,
        "right_hand_motor_q": right_hand_motor_q,
        "right_hand_motor_dq": right_hand_motor_dq,
        "right_hand_motor_kp": right_hand_motor_kp,
        "right_hand_motor_kd": right_hand_motor_kd,
        "right_hand_motor_tau": right_hand_motor_tau,
        "right_hand_motor_tau_est": right_hand_motor_tau_est,
        "mujoco_qpos": mujoco_qpos,
        "mujoco_qvel": mujoco_qvel,
        "substep_time": substep_time,
        "substep_qpos": substep_qpos,
        "substep_qvel": substep_qvel,
        "substep_ctrl": substep_ctrl,
        "substep_qfrc": substep_qfrc,
        "substep_xfrc": substep_xfrc,
        "substep_warm": substep_warm,
    }


def _reconstruct_body_q_target_from_raw_action(action43: np.ndarray) -> np.ndarray:
    """Convert recorded raw policy body action to deploy-style MuJoCo q target."""
    raw_body29 = action43[_BODY_IDX]
    return _DEFAULT_ANGLES + raw_body29[_ISAACLAB_TO_MUJOCO] * _ACTION_SCALE


def _body_q_target_from_action(action43: np.ndarray, body_action_format: str) -> np.ndarray:
    if body_action_format == "q_target":
        return action43[_BODY_IDX].copy()
    if body_action_format == "raw_policy":
        return _reconstruct_body_q_target_from_raw_action(action43)
    raise ValueError(f"Unknown body action format: {body_action_format!r}")


def _apply_elastic_band(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    elastic_band: ElasticBand,
    body_id: int,
) -> None:
    """Match BaseSim.sim_step's external elastic-band root support."""
    pose = np.concatenate(
        [
            data.xpos[body_id],
            data.xquat[body_id],
            np.zeros(6),
        ]
    )
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        pose[7:13],
        0,
    )
    pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
    data.xfrc_applied[body_id] = elastic_band.Advance(pose)


def _infer_body_action_format(
    actions: np.ndarray, states: np.ndarray, max_frames: int = 200
) -> tuple[str, float, float]:
    """Infer whether action.wbc body values are already q targets or raw actions."""
    n = min(max_frames, len(actions), len(states))
    if n == 0:
        return "q_target", float("nan"), float("nan")
    state_body = states[:n, _BODY_IDX]
    action_body = actions[:n, _BODY_IDX]
    recon_body = np.stack(
        [_reconstruct_body_q_target_from_raw_action(a) for a in actions[:n]]
    )
    raw_rmse = float(np.sqrt(np.mean((action_body - state_body) ** 2)))
    recon_rmse = float(np.sqrt(np.mean((recon_body - state_body) ** 2)))
    inferred = "raw_policy" if recon_rmse < raw_rmse * 0.75 else "q_target"
    return inferred, raw_rmse, recon_rmse


def sim_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ctrl_ids: np.ndarray,
    body29_q_des: np.ndarray,
    body29_dq_des: np.ndarray,
    body29_tau_ff: np.ndarray,
    body29_kp: np.ndarray,
    body29_kd: np.ndarray,
    left_ctrl_ids: np.ndarray,
    left7_target: np.ndarray,
    right_ctrl_ids: np.ndarray,
    right7_target: np.ndarray,
    left7_dq_des: np.ndarray | None = None,
    left7_tau_ff: np.ndarray | None = None,
    left7_kp: np.ndarray | None = None,
    left7_kd: np.ndarray | None = None,
    right7_dq_des: np.ndarray | None = None,
    right7_tau_ff: np.ndarray | None = None,
    right7_kp: np.ndarray | None = None,
    right7_kd: np.ndarray | None = None,
    left7_is_lowcmd: bool = False,
    right7_is_lowcmd: bool = False,
    body29_direct_tau: np.ndarray | None = None,
    left7_direct_tau: np.ndarray | None = None,
    right7_direct_tau: np.ndarray | None = None,
    n_steps: int = _SIM_STEPS_PER_POLICY,
    root_jid: int = -1,
    base_pos_start: np.ndarray | None = None,
    base_pos_end: np.ndarray | None = None,
    base_quat_start: np.ndarray | None = None,
    base_quat_end: np.ndarray | None = None,
    elastic_band: ElasticBand | None = None,
    elastic_band_body_id: int = -1,
    pd_once_per_frame: bool = False,
) -> None:
    """As ``base_sim.compute_body_torques`` + hands PD + ``mj_step``, with optional
    root kinematic drive (same as ``playback_lerobot._physics_step``)."""
    drive_base = (
        root_jid >= 0
        and base_pos_start is not None
        and base_pos_end is not None
        and base_quat_start is not None
        and base_quat_end is not None
    )

    qpos_adr = qvel_adr = 0
    lin_vel = ang_vel_world = np.zeros(3)

    if drive_base:
        qpos_adr = model.jnt_qposadr[root_jid]
        qvel_adr = model.jnt_dofadr[root_jid]
        policy_dt = n_steps * _SIM_DT
        lin_vel = (base_pos_end - base_pos_start) / policy_dt  # type: ignore[operator]
        q_diff = _quat_mult(_quat_conj(base_quat_start), base_quat_end)  # type: ignore[arg-type]
        half_angle = float(np.arccos(np.clip(abs(float(q_diff[0])), 0.0, 1.0)))
        if half_angle < 1e-8:
            ang_vel_world = np.zeros(3)
        else:
            axis_body = q_diff[1:4] / np.sin(half_angle)
            R = _quat_to_rot_matrix(base_quat_start)  # type: ignore[arg-type]
            ang_vel_world = R @ (axis_body * (2.0 * half_angle / policy_dt))

    frame_ctrl: np.ndarray | None = None
    for k in range(n_steps):
        if drive_base:
            frac = k / n_steps
            pos_k = (1.0 - frac) * base_pos_start + frac * base_pos_end  # type: ignore[operator]
            quat_k = _quat_slerp(base_quat_start, base_quat_end, frac)  # type: ignore[arg-type]
            data.qpos[qpos_adr : qpos_adr + 3] = pos_k
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_k
            data.qvel[qvel_adr : qvel_adr + 3] = lin_vel
            data.qvel[qvel_adr + 3 : qvel_adr + 6] = ang_vel_world

        if elastic_band is not None and elastic_band_body_id >= 0:
            _apply_elastic_band(model, data, elastic_band, elastic_band_body_id)
        elif elastic_band_body_id >= 0:
            data.xfrc_applied[elastic_band_body_id] = np.zeros(6)

        if pd_once_per_frame and frame_ctrl is not None:
            data.ctrl[:] = frame_ctrl
            mujoco.mj_step(model, data)
            continue

        ctrl = np.zeros(model.nu)

        for j, cid in enumerate(body_ctrl_ids):
            if cid < 0:
                continue
            tlim = _TORQUE_LIMIT_BODY29[j]
            if body29_direct_tau is not None:
                tau = body29_direct_tau[j]
            else:
                ajid = model.actuator(cid).trnid[0]
                q = data.qpos[model.jnt_qposadr[ajid]]
                dq = data.qvel[model.jnt_dofadr[ajid]]
                tau = (
                    body29_tau_ff[j]
                    + body29_kp[j] * (body29_q_des[j] - q)
                    + body29_kd[j] * (body29_dq_des[j] - dq)
                )
            ctrl[cid] = np.clip(tau, -tlim, tlim)

        for (
            target_arr,
            cids,
            max_limits,
            min_limits,
            dq_des,
            tau_ff,
            hand_kp,
            hand_kd,
            is_lowcmd,
            direct_tau,
        ) in (
            (
                left7_target,
                left_ctrl_ids,
                _HAND_MAX_LIMITS_LEFT,
                _HAND_MIN_LIMITS_LEFT,
                left7_dq_des,
                left7_tau_ff,
                left7_kp,
                left7_kd,
                left7_is_lowcmd,
                left7_direct_tau,
            ),
            (
                right7_target,
                right_ctrl_ids,
                _HAND_MAX_LIMITS_RIGHT,
                _HAND_MIN_LIMITS_RIGHT,
                right7_dq_des,
                right7_tau_ff,
                right7_kp,
                right7_kd,
                right7_is_lowcmd,
                right7_direct_tau,
            ),
        ):
            if direct_tau is not None:
                for j, cid in enumerate(cids):
                    if cid >= 0:
                        ctrl[cid] = np.clip(
                            direct_tau[j], -_TORQUE_LIMIT_HAND, _TORQUE_LIMIT_HAND
                        )
                continue
            if dq_des is None:
                dq_des = np.zeros(len(cids), dtype=np.float64)
            if tau_ff is None:
                tau_ff = np.zeros(len(cids), dtype=np.float64)
            if hand_kp is None:
                hand_kp = np.full(len(cids), _HAND_KP_DEFAULT, dtype=np.float64)
            if hand_kd is None:
                hand_kd = np.full(len(cids), _HAND_KD_DEFAULT, dtype=np.float64)
            current_q = np.zeros(len(cids), dtype=np.float64)
            for j, cid in enumerate(cids):
                if cid >= 0:
                    ajid = model.actuator(cid).trnid[0]
                    current_q[j] = data.qpos[model.jnt_qposadr[ajid]]
            smoothed_target = (
                target_arr
                if is_lowcmd
                else _dex3_smoothed_hand_target(target_arr, current_q, max_limits, min_limits)
            )
            for j, cid in enumerate(cids):
                if cid < 0:
                    continue
                ajid = model.actuator(cid).trnid[0]
                q = data.qpos[model.jnt_qposadr[ajid]]
                dq = data.qvel[model.jnt_dofadr[ajid]]
                tau = tau_ff[j] + hand_kp[j] * (smoothed_target[j] - q) + hand_kd[j] * (
                    dq_des[j] - dq
                )
                ctrl[cid] = np.clip(tau, -_TORQUE_LIMIT_HAND, _TORQUE_LIMIT_HAND)

        data.ctrl[:] = ctrl
        if pd_once_per_frame:
            frame_ctrl = ctrl.copy()
        mujoco.mj_step(model, data)

    if drive_base:
        data.qpos[qpos_adr : qpos_adr + 3] = base_pos_end  # type: ignore[index]
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = base_quat_end  # type: ignore[index]
        data.qvel[qvel_adr : qvel_adr + 3] = lin_vel
        data.qvel[qvel_adr + 3 : qvel_adr + 6] = ang_vel_world


def _task_description(dataset_dir: str, task_index: int | None) -> str:
    tasks_file = os.path.join(dataset_dir, "meta/tasks.jsonl")
    if task_index is None or not os.path.exists(tasks_file):
        return "(unknown)"
    with open(tasks_file) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("task_index") == task_index:
                return rec.get("task", "(unknown)")
    return "(unknown)"


def playback_action_pd(
    dataset_dir: str,
    episode: int,
    env_name: str,
    fps: float,
    no_viewer: bool,
    sim_substeps: int | None,
    free_base: bool,
    debug: bool,
    gains_yaml: str | None = None,
    hold_recorded_waist: bool = False,
    body_action_format: str = "auto",
    use_action_q_cmd: bool = False,
    use_recorded_torque: bool = False,
    drive_recorded_base: bool = False,
    elastic_band: bool = False,
    elastic_band_body: str = "auto",
    pd_once_per_frame: bool = False,
    use_recorded_substep_inputs: bool = False,
    init_substep_frame: int = 0,
    init_substep_index: int = 0,
    init_substep_all_inputs: bool = False,
    start_frame: int = 0,
    output_video: str | None = None,
    video_width: int = 1280,
    video_height: int = 720,
) -> None:
    rec = _load_action_episode(dataset_dir, episode)
    states = rec["states"]
    actions = rec["actions"]
    task_indices = rec["task_indices"]
    base_pos = rec["base_pos"]
    base_quat = rec["base_quat"]
    q_cmd = rec["q_target_cmd"]
    motor_q = rec["motor_q"]
    motor_kp = rec["motor_kp"]
    motor_kd = rec["motor_kd"]
    motor_tau = rec["motor_tau"]
    motor_tau_est = rec["motor_tau_est"]
    motor_dq = rec["motor_dq"]
    left_hand_motor_q = rec["left_hand_motor_q"]
    left_hand_motor_dq = rec["left_hand_motor_dq"]
    left_hand_motor_kp = rec["left_hand_motor_kp"]
    left_hand_motor_kd = rec["left_hand_motor_kd"]
    left_hand_motor_tau = rec["left_hand_motor_tau"]
    left_hand_motor_tau_est = rec["left_hand_motor_tau_est"]
    right_hand_motor_q = rec["right_hand_motor_q"]
    right_hand_motor_dq = rec["right_hand_motor_dq"]
    right_hand_motor_kp = rec["right_hand_motor_kp"]
    right_hand_motor_kd = rec["right_hand_motor_kd"]
    right_hand_motor_tau = rec["right_hand_motor_tau"]
    right_hand_motor_tau_est = rec["right_hand_motor_tau_est"]
    mujoco_qpos = rec["mujoco_qpos"]
    mujoco_qvel = rec["mujoco_qvel"]
    substep_time = rec["substep_time"]
    substep_qpos = rec["substep_qpos"]
    substep_qvel = rec["substep_qvel"]
    substep_ctrl = rec["substep_ctrl"]
    substep_qfrc = rec["substep_qfrc"]
    substep_xfrc = rec["substep_xfrc"]
    substep_warm = rec["substep_warm"]

    total_frames = len(actions)
    task_index = task_indices[0] if task_indices else None

    yaml_kp, yaml_kd, yaml_err = _load_motor_gains_wbc_yaml(gains_yaml)
    selected_body_action_format = body_action_format
    body_action_raw_rmse = float("nan")
    body_action_recon_rmse = float("nan")
    if (use_action_q_cmd or (motor_q is None and q_cmd is None)) and selected_body_action_format == "auto":
        (
            selected_body_action_format,
            body_action_raw_rmse,
            body_action_recon_rmse,
        ) = _infer_body_action_format(actions, states)
    if use_recorded_torque and motor_tau_est is None:
        raise ValueError(
            "--use_recorded_torque requires 'robot.motor_tau_est' in the dataset. "
            "Record with the lowcmd debug exporter after rebuilding/restarting deploy."
        )

    model = _load_xml(env_name)
    model.opt.timestep = _SIM_DT
    effective_sim_substeps = (
        sim_substeps
        if sim_substeps is not None
        else max(1, int(round((1.0 / fps) / _SIM_DT)))
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    root_jids = [i for i in range(model.njnt) if model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE]
    root_jid = root_jids[0] if root_jids else None

    body_jids, left_jids, right_jids = _build_joint_indices(model)
    body_ctrl_ids = _build_ctrl_map(model, body_jids)
    left_ctrl_ids = _build_ctrl_map(model, left_jids)
    right_ctrl_ids = _build_ctrl_map(model, right_jids)

    def reshape_substeps(arr: np.ndarray | None, width: int, name: str) -> np.ndarray:
        if arr is None:
            raise ValueError(f"--use_recorded_substep_inputs requires {name} in the dataset.")
        if arr.shape[1] % width != 0:
            raise ValueError(f"{name} width {arr.shape[1]} is not divisible by {width}.")
        return arr.reshape(arr.shape[0], arr.shape[1] // width, width)

    recorded_substeps = 0
    frame_time = frame_qpos = frame_qvel = frame_ctrl = frame_qfrc = frame_warm = None
    frame_xfrc = None
    if use_recorded_substep_inputs:
        if substep_time is None:
            raise ValueError("--use_recorded_substep_inputs requires robot.motor_pd_substep_sim_time.")
        frame_time = substep_time
        frame_qpos = reshape_substeps(substep_qpos, model.nq, "robot.mujoco_substep_qpos")
        frame_qvel = reshape_substeps(substep_qvel, model.nv, "robot.mujoco_substep_qvel")
        frame_ctrl = reshape_substeps(substep_ctrl, model.nu, "robot.mujoco_substep_ctrl")
        frame_qfrc = reshape_substeps(substep_qfrc, model.nv, "robot.mujoco_substep_qfrc_applied")
        frame_warm = reshape_substeps(substep_warm, model.nv, "robot.mujoco_substep_qacc_warmstart")
        frame_xfrc = reshape_substeps(
            substep_xfrc,
            model.nbody * 6,
            "robot.mujoco_substep_xfrc_applied",
        ).reshape(total_frames, -1, model.nbody, 6)
        recorded_substeps = min(
            frame_time.shape[1],
            frame_qpos.shape[1],
            frame_qvel.shape[1],
            frame_ctrl.shape[1],
            frame_qfrc.shape[1],
            frame_xfrc.shape[1],
            frame_warm.shape[1],
        )
    elastic_band_obj = ElasticBand() if elastic_band else None
    elastic_band_body_id = -1
    elastic_band_body_name = ""
    if elastic_band_obj is not None:
        body_candidates = (
            ("pelvis", "torso_link") if elastic_band_body == "auto" else (elastic_band_body,)
        )
        for body_name in body_candidates:
            try:
                elastic_band_body_id = model.body(body_name).id
                elastic_band_body_name = body_name
                break
            except Exception:
                continue
        if elastic_band_body_id < 0:
            raise ValueError(
                f"Could not attach elastic band to {elastic_band_body!r}; "
                "expected a body named 'pelvis' or 'torso_link'."
            )

    init_from_recorded_mujoco = (
        mujoco_qpos is not None
        and mujoco_qvel is not None
        and mujoco_qpos.shape[1] <= model.nq
        and mujoco_qvel.shape[1] <= model.nv
    )
    init_mujoco_full_scene = (
        init_from_recorded_mujoco
        and mujoco_qpos is not None
        and mujoco_qvel is not None
        and mujoco_qpos.shape[1] == model.nq
        and mujoco_qvel.shape[1] == model.nv
    )
    if use_recorded_substep_inputs:
        assert frame_time is not None
        assert frame_qpos is not None
        assert frame_qvel is not None
        assert frame_ctrl is not None
        assert frame_qfrc is not None
        assert frame_xfrc is not None
        assert frame_warm is not None
        init_substep_frame = int(np.clip(init_substep_frame, 0, total_frames - 1))
        init_substep_index = int(np.clip(init_substep_index, 0, recorded_substeps - 1))
        if not np.isfinite(frame_qpos[init_substep_frame, init_substep_index]).all():
            raise ValueError(
                f"Recorded qpos is not finite at f{init_substep_frame}s{init_substep_index}."
            )
        data.qpos[:] = frame_qpos[init_substep_frame, init_substep_index]
        data.qvel[:] = frame_qvel[init_substep_frame, init_substep_index]
        init_time = float(frame_time[init_substep_frame, init_substep_index])
        if np.isfinite(init_time):
            data.time = init_time
        if init_substep_all_inputs:
            data.ctrl[:] = frame_ctrl[init_substep_frame, init_substep_index]
            data.qfrc_applied[:] = frame_qfrc[init_substep_frame, init_substep_index]
            data.xfrc_applied[:] = frame_xfrc[init_substep_frame, init_substep_index]
            data.qacc_warmstart[:] = frame_warm[init_substep_frame, init_substep_index]
    elif init_from_recorded_mujoco:
        data.qpos[: mujoco_qpos.shape[1]] = mujoco_qpos[0]  # type: ignore[index]
        data.qvel[: mujoco_qvel.shape[1]] = mujoco_qvel[0]  # type: ignore[index]
    else:
        init_base_pos = (
            base_pos[0] if base_pos is not None else np.array([0.0, 0.0, _ROOT_HEIGHT], dtype=np.float64)
        )
        init_base_quat = (
            base_quat[0] if base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        _set_qpos(
            data,
            model,
            body_jids,
            left_jids,
            right_jids,
            states[0],
            root_jid=root_jid,
            base_pos=init_base_pos,
            base_quat=init_base_quat,
        )
    mujoco.mj_forward(model, data)

    def src(tag: str, have: bool, fallback: str) -> str:
        return f"{tag}: {'per-frame Parquet' if have else fallback}"

    print(f"Episode {episode}: {total_frames} action frames @ {fps:.0f} Hz")
    print(f"Task [{task_index}]: {_task_description(dataset_dir, task_index)}")
    print(f"Environment: {env_name}")
    print(f"MuJoCo timestep: {model.opt.timestep:g}s, substeps/frame: {effective_sim_substeps}")
    print(
        "PD update: "
        f"{'once per recorded frame, held through substeps' if pd_once_per_frame else 'every MuJoCo substep'}"
    )
    wbc_path = str(Path(gains_yaml) if gains_yaml else _wbc_yaml_default())
    k_fallback = f"WBC yaml ({wbc_path})" if yaml_err is None else "C++ armature defaults (playback_lerobot)"
    if yaml_err and gains_yaml is None:
        k_fallback = f"WBC yaml missing ({yaml_err}) → C++ armature defaults"
    elif yaml_err and gains_yaml is not None:
        k_fallback = f"WBC yaml invalid ({yaml_err}) → C++ armature defaults"
    print(
        src(
            "Body q_cmd",
            (not use_action_q_cmd) and (motor_q is not None or q_cmd is not None),
            (
                "sonic.q_target_cmd"
                if q_cmd is not None
                else f"action.wbc body ({selected_body_action_format})"
            ),
        )
    )
    if (use_action_q_cmd or (motor_q is None and q_cmd is None)) and body_action_format == "auto":
        print(
            "Body action auto-detect: "
            f"raw_vs_state_rmse={body_action_raw_rmse:.4f}, "
            f"reconstructed_vs_state_rmse={body_action_recon_rmse:.4f} "
            f"-> {selected_body_action_format}"
        )
    print(src("Kp", motor_kp is not None, k_fallback))
    print(src("Kd", motor_kd is not None, k_fallback))
    print(src("tau_ff", motor_tau is not None, "zeros"))
    if use_recorded_torque:
        print("Playback mode: recorded torque (robot.motor_tau_est -> body actuators; no body PD/action/lowcmd q)")
        if left_hand_motor_tau_est is None:
            print("Left hand recorded torque: missing; hand actuator torques set to zero")
        else:
            print("Left hand recorded torque: robot.left_hand_motor_tau_est")
        if right_hand_motor_tau_est is None:
            print("Right hand recorded torque: missing; hand actuator torques set to zero")
        else:
            print("Right hand recorded torque: robot.right_hand_motor_tau_est")
    print(src("dq_des (cmd)", motor_dq is not None, "zeros"))
    print(src("Left hand q_cmd", (not use_action_q_cmd) and left_hand_motor_q is not None, "action.wbc left hand"))
    print(src("Right hand q_cmd", (not use_action_q_cmd) and right_hand_motor_q is not None, "action.wbc right hand"))
    if use_recorded_substep_inputs:
        print(
            "Playback mode: recorded MuJoCo substep inputs "
            "(qpos/qvel/ctrl/qfrc/xfrc/qacc_warmstart -> mj_step)"
        )
        print(
            f"Initial state: recorded substep f{init_substep_frame}s{init_substep_index} "
            f"({'all inputs' if init_substep_all_inputs else 'qpos/qvel/time only'})"
        )
    elif init_from_recorded_mujoco:
        qpos_shape = mujoco_qpos.shape[1] if mujoco_qpos is not None else None
        qvel_shape = mujoco_qvel.shape[1] if mujoco_qvel is not None else None
        scene_note = "full scene" if init_mujoco_full_scene else "recorded prefix"
        print(f"Initial state: recorded robot.mujoco_qpos/qvel ({scene_note}; {qpos_shape}/{qvel_shape})")
    else:
        qpos_shape = None if mujoco_qpos is None else mujoco_qpos.shape[1]
        qvel_shape = None if mujoco_qvel is None else mujoco_qvel.shape[1]
        print(
            "Initial state: observation.state + base pose "
            f"(no matching full qpos/qvel; got qpos={qpos_shape}, qvel={qvel_shape}, "
            f"model nq/nv={model.nq}/{model.nv})"
        )
    base_drive_enabled = drive_recorded_base or not free_base
    print(f"Base: {'driven from recording when available' if base_drive_enabled else 'free'}")
    print(
        "Elastic band: "
        f"{'enabled on ' + elastic_band_body_name if elastic_band_obj is not None else 'disabled'}"
    )
    print(
        "Waist targets: "
        f"{'recorded observation.state' if hold_recorded_waist else 'action body target'}"
    )
    print(
        f"Actuators - body: {(body_ctrl_ids >= 0).sum()}/{len(body_jids)}, "
        f"left_hand: {(left_ctrl_ids >= 0).sum()}/{len(left_jids)}, "
        f"right_hand: {(right_ctrl_ids >= 0).sum()}/{len(right_jids)}"
    )

    viewer = None
    renderer = None
    video_writer = None
    if not no_viewer:
        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
        if viewer is not None:
            try:
                pelvis_id = model.body("pelvis").id
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = pelvis_id
                viewer.cam.distance = 2.5
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 135
            except Exception:
                pass
    if output_video is not None:
        output_path = Path(output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=video_height, width=video_width)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_width, video_height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open output video writer: {output_path}")
        print(f"Saving playback video to {output_path}")

    def write_video_frame() -> None:
        if renderer is None or video_writer is None:
            return
        renderer.update_scene(data)
        rgb = renderer.render()
        video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    zero29 = np.zeros(29, dtype=np.float64)

    dt = 1.0 / fps
    try:
        for i in range(start_frame, total_frames):
            action = actions[i]
            t_start = time.perf_counter()

            if use_recorded_substep_inputs:
                assert frame_time is not None
                assert frame_qpos is not None
                assert frame_qvel is not None
                assert frame_ctrl is not None
                assert frame_qfrc is not None
                assert frame_xfrc is not None
                assert frame_warm is not None

                def replay_recorded_substeps() -> None:
                    for substep in range(recorded_substeps):
                        substep_time = float(frame_time[i, substep])
                        if not np.isfinite(substep_time):
                            continue
                        if substep_time < data.time - _SUBSTEP_TIME_EPS:
                            continue
                        qpos = frame_qpos[i, substep]
                        qvel = frame_qvel[i, substep]
                        ctrl = frame_ctrl[i, substep]
                        qfrc = frame_qfrc[i, substep]
                        xfrc = frame_xfrc[i, substep]
                        warm = frame_warm[i, substep]
                        if not (
                            np.isfinite(qpos).all()
                            and np.isfinite(qvel).all()
                            and np.isfinite(ctrl).all()
                            and np.isfinite(qfrc).all()
                            and np.isfinite(xfrc).all()
                            and np.isfinite(warm).all()
                        ):
                            continue
                        data.time = substep_time
                        data.qpos[:] = qpos
                        data.qvel[:] = qvel
                        data.ctrl[:] = ctrl
                        data.qfrc_applied[:] = qfrc
                        data.xfrc_applied[:] = xfrc
                        data.qacc_warmstart[:] = warm
                        mujoco.mj_step(model, data)

                if viewer is not None:
                    with viewer.lock():
                        replay_recorded_substeps()
                else:
                    replay_recorded_substeps()

                if viewer is not None and viewer.is_running():
                    viewer.sync()
                write_video_frame()
                if debug and (i < start_frame + 10 or (i + 1) % 100 == 0 or i == total_frames - 1):
                    print(f"  Frame {i + 1}/{total_frames}  time={data.time:.6f}")
                elif i < start_frame + 10 or (i + 1) % 100 == 0 or i == total_frames - 1:
                    print(f"  Frame {i + 1}/{total_frames}")
                elapsed = time.perf_counter() - t_start
                time.sleep(max(0.0, dt - elapsed))
                continue

            if use_action_q_cmd:
                body29_q_des = _body_q_target_from_action(
                    action, selected_body_action_format
                )
            elif motor_q is not None:
                body29_q_des = motor_q[i].copy()
            elif q_cmd is not None:
                body29_q_des = q_cmd[i].copy()
            else:
                body29_q_des = _body_q_target_from_action(
                    action, selected_body_action_format
                )
            if hold_recorded_waist:
                body29_q_des[_WAIST_SLICE_BODY29] = states[i][_BODY_IDX][
                    _WAIST_SLICE_BODY29
                ]
            if motor_kp is not None:
                kpb = motor_kp[i]
            else:
                kpb = yaml_kp
            if motor_kd is not None:
                kdb = motor_kd[i]
            else:
                kdb = yaml_kd
            if motor_tau is not None:
                tau_b = motor_tau[i]
            else:
                tau_b = zero29
            if motor_dq is not None:
                dq_b = motor_dq[i]
            else:
                dq_b = zero29
            direct_body_tau = motor_tau_est[i] if use_recorded_torque else None
            direct_left_tau = None
            direct_right_tau = None
            if use_recorded_torque:
                direct_left_tau = (
                    left_hand_motor_tau_est[i]
                    if left_hand_motor_tau_est is not None
                    else np.zeros(7, dtype=np.float64)
                )
                direct_right_tau = (
                    right_hand_motor_tau_est[i]
                    if right_hand_motor_tau_est is not None
                    else np.zeros(7, dtype=np.float64)
                )

            left7_target = (
                left_hand_motor_q[i]
                if (not use_action_q_cmd) and left_hand_motor_q is not None
                else action[_LEFT_HAND_IDX]
            )
            right7_target = (
                right_hand_motor_q[i]
                if (not use_action_q_cmd) and right_hand_motor_q is not None
                else action[_RIGHT_HAND_IDX]
            )

            i_next = min(i + 1, total_frames - 1)
            use_base_drive = base_drive_enabled and base_pos is not None and base_quat is not None
            def step_frame() -> None:
                sim_step(
                    model,
                    data,
                    body_ctrl_ids,
                    body29_q_des,
                    dq_b,
                    tau_b,
                    kpb,
                    kdb,
                    left_ctrl_ids,
                    left7_target,
                    right_ctrl_ids,
                    right7_target,
                    left7_dq_des=left_hand_motor_dq[i] if left_hand_motor_dq is not None else None,
                    left7_tau_ff=left_hand_motor_tau[i] if left_hand_motor_tau is not None else None,
                    left7_kp=left_hand_motor_kp[i] if left_hand_motor_kp is not None else None,
                    left7_kd=left_hand_motor_kd[i] if left_hand_motor_kd is not None else None,
                    right7_dq_des=right_hand_motor_dq[i] if right_hand_motor_dq is not None else None,
                    right7_tau_ff=right_hand_motor_tau[i] if right_hand_motor_tau is not None else None,
                    right7_kp=right_hand_motor_kp[i] if right_hand_motor_kp is not None else None,
                    right7_kd=right_hand_motor_kd[i] if right_hand_motor_kd is not None else None,
                    left7_is_lowcmd=(not use_action_q_cmd) and left_hand_motor_q is not None,
                    right7_is_lowcmd=(not use_action_q_cmd) and right_hand_motor_q is not None,
                    body29_direct_tau=direct_body_tau,
                    left7_direct_tau=direct_left_tau,
                    right7_direct_tau=direct_right_tau,
                    n_steps=effective_sim_substeps,
                    root_jid=root_jid if use_base_drive and root_jid is not None else -1,
                    base_pos_start=base_pos[i] if use_base_drive else None,
                    base_pos_end=base_pos[i_next] if use_base_drive else None,
                    base_quat_start=base_quat[i] if use_base_drive else None,
                    base_quat_end=base_quat[i_next] if use_base_drive else None,
                    elastic_band=elastic_band_obj,
                    elastic_band_body_id=elastic_band_body_id,
                    pd_once_per_frame=pd_once_per_frame,
                )
                # mujoco.mj_forward(model, data)

            if viewer is not None:
                with viewer.lock():
                    step_frame()
            else:
                step_frame()

            if viewer is not None and viewer.is_running():
                viewer.sync()
            write_video_frame()

            if debug and (i < 10 or (i + 1) % 100 == 0 or i == total_frames - 1):
                sim_body29 = data.qpos[model.jnt_qposadr[body_jids]]
                root_msg = ""
                if use_base_drive and root_jid is not None:
                    root_adr = model.jnt_qposadr[root_jid]
                    root_msg = (
                        f" base={np.round(data.qpos[root_adr : root_adr + 3], 4)}"
                        f" rec_base={np.round(base_pos[i_next], 4)}"
                        f" quat={np.round(data.qpos[root_adr + 3 : root_adr + 7], 4)}"
                    )
                command_msg = (
                    f"tau_leg={np.round(direct_body_tau[:12], 4)}"
                    if direct_body_tau is not None
                    else f"q_des_leg={np.round(body29_q_des[:12], 4)}"
                )
                print(
                    f"  Frame {i + 1}/{total_frames} "
                    f"{command_msg} "
                    f"sim_leg={np.round(sim_body29[:12], 4)}"
                    f"{root_msg}"
                )
            elif i < 10 or (i + 1) % 100 == 0 or i == total_frames - 1:
                print(f"  Frame {i + 1}/{total_frames}")

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, dt - elapsed))
    finally:
        if viewer is not None:
            viewer.close()
        if video_writer is not None:
            video_writer.release()
        if renderer is not None:
            renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay LeRobot action through teleop-style PD (see sim_step in this file)."
    )
    parser.add_argument("--dataset_dir", required=True, help="Path to LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument(
        "--env_name",
        default="default",
        choices=["kitchen_pnp_apple", "pnp_cube", "lift_box", "pnp_bottle", "default"],
        help="MuJoCo scene to load.",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Playback rate in Hz.")
    parser.add_argument("--no_viewer", action="store_true", help="Disable the on-screen viewer.")
    parser.add_argument(
        "--sim_substeps",
        type=int,
        default=None,
        help=(
            "MuJoCo substeps per recorded action frame. Default derives from "
            "fps and SIMULATE_DT to match teleop timing."
        ),
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Compatibility flag; this script always uses MuJoCo physics.",
    )
    parser.add_argument(
        "--free_base",
        action="store_true",
        help="Do not drive robot.base_pos/robot.base_quat; let the floating base simulate freely.",
    )
    parser.add_argument(
        "--drive_recorded_base",
        action="store_true",
        help="Compatibility flag; recorded base is driven by default unless --free_base is set.",
    )
    parser.add_argument(
        "--gains-yaml",
        default=None,
        help=(
            "YAML with MOTOR_KP / MOTOR_KD (e.g. g1_29dof_gear_wbc.yaml). "
            "Default: gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml."
        ),
    )
    parser.add_argument(
        "--hold_recorded_waist",
        action="store_true",
        help=(
            "Use recorded observation.state for waist_yaw/roll/pitch targets instead "
            "of action.wbc. Pelvis/root orientation is still driven by recorded base_quat."
        ),
    )
    parser.add_argument(
        "--body_action_format",
        choices=_BODY_ACTION_FORMATS,
        default="auto",
        help=(
            "How to interpret action.wbc body values when sonic.q_target_cmd is missing. "
            "'q_target' uses action.wbc directly; 'raw_policy' applies deploy "
            "default_angles + raw_action[isaaclab_to_mujoco] * g1_action_scale; "
            "'auto' chooses the format closer to recorded observation.state."
        ),
    )
    parser.add_argument(
        "--use_action_q_cmd",
        action="store_true",
        help=(
            "Use action.wbc for body/hand q commands while still using recorded "
            "low-level kp/kd/dq/tau when available."
        ),
    )
    parser.add_argument(
        "--use_recorded_torque",
        action="store_true",
        help=(
            "Replay recorded real-time torque directly from robot.motor_tau_est. "
            "This bypasses body PD and does not use action.wbc or lowcmd q/dq/kp/kd/tau "
            "to compute body actuator torques. Hand tau_est is used when present; "
            "otherwise hand torques are zero."
        ),
    )
    parser.add_argument(
        "--pd_once_per_frame",
        action="store_true",
        help=(
            "Compute PD torque once at each recorded frame boundary and hold that "
            "same ctrl through MuJoCo substeps. Default recomputes PD every substep."
        ),
    )
    parser.add_argument(
        "--use_recorded_substep_inputs",
        action="store_true",
        help=(
            "Replay recorded MuJoCo substep inputs directly from robot.mujoco_substep_* "
            "columns instead of recomputing PD."
        ),
    )
    parser.add_argument(
        "--init_substep_frame",
        type=int,
        default=0,
        help="Frame index used to initialize recorded-substep playback.",
    )
    parser.add_argument(
        "--init_substep_index",
        type=int,
        default=0,
        help="Substep index within --init_substep_frame used for initialization.",
    )
    parser.add_argument(
        "--init_substep_all_inputs",
        action="store_true",
        help="Also initialize ctrl/qfrc/xfrc/qacc_warmstart from the selected substep.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First recorded frame to play back.",
    )
    parser.add_argument(
        "--output_video",
        default=None,
        help="Optional MP4 path. If set, records offscreen playback frames to this file.",
    )
    parser.add_argument("--video_width", type=int, default=1280, help="Output video width.")
    parser.add_argument("--video_height", type=int, default=720, help="Output video height.")
    parser.add_argument(
        "--elastic_band",
        action="store_true",
        help="Apply the same ElasticBand external root support force used by live MuJoCo sim.",
    )
    parser.add_argument(
        "--elastic_band_body",
        choices=_ELASTIC_BAND_BODIES,
        default="auto",
        help="Body to attach ElasticBand to. 'auto' prefers pelvis, then torso_link.",
    )
    parser.add_argument("--debug", action="store_true", help="Print target and simulated leg qpos.")
    args = parser.parse_args()

    playback_action_pd(
        dataset_dir=args.dataset_dir,
        episode=args.episode,
        env_name=args.env_name,
        fps=args.fps,
        no_viewer=args.no_viewer,
        sim_substeps=args.sim_substeps,
        free_base=args.free_base,
        debug=args.debug,
        gains_yaml=args.gains_yaml,
        hold_recorded_waist=args.hold_recorded_waist,
        body_action_format=args.body_action_format,
        use_action_q_cmd=args.use_action_q_cmd,
        use_recorded_torque=args.use_recorded_torque,
        drive_recorded_base=args.drive_recorded_base,
        elastic_band=args.elastic_band,
        elastic_band_body=args.elastic_band_body,
        pd_once_per_frame=args.pd_once_per_frame,
        use_recorded_substep_inputs=args.use_recorded_substep_inputs,
        init_substep_frame=args.init_substep_frame,
        init_substep_index=args.init_substep_index,
        init_substep_all_inputs=args.init_substep_all_inputs,
        start_frame=args.start_frame,
        output_video=args.output_video,
        video_width=args.video_width,
        video_height=args.video_height,
    )


if __name__ == "__main__":
    main()
