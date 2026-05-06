"""ONNX playback that rebuilds encoder input from recorded future teleop frames.

The encoder future fields use 10 continuous frames: current frame plus the next
9 frames.  Near the episode end, where that full future window is unavailable,
this script falls back to the recorded ``sonic.encoder_obs`` for the whole
encoder input.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq

from playback_lerobot import (
    _BODY_IDX,
    _ENC_OBS_LAYOUT,
    _LEFT_HAND_IDX,
    _RIGHT_HAND_IDX,
    _SIM_DT,
    SonicRunner,
    _build_ctrl_map,
    _build_joint_indices,
    _load_episode,
    _load_xml,
    _physics_step,
    _set_qpos,
)
from playback_lerobot_onnx_inputs_pd import _sim_state43


def _episode_path(dataset_dir: str, episode: int) -> str:
    chunk = episode // 1000
    return f"{dataset_dir}/data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"


def _load_required_array(table: pq.Table, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"Dataset is missing required column {name!r}.")
    return np.array([row.as_py() for row in table.column(name)], dtype=np.float32)


def _load_future_fields(dataset_dir: str, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _episode_path(dataset_dir, episode)
    table = pq.read_table(
        path,
        columns=[
            "teleop.smpl_joints",
            "teleop.left_wrist_joints",
            "teleop.right_wrist_joints",
        ],
    )
    smpl_joints = _load_required_array(table, "teleop.smpl_joints")
    left_wrist = _load_required_array(table, "teleop.left_wrist_joints")
    right_wrist = _load_required_array(table, "teleop.right_wrist_joints")

    if smpl_joints.shape[1] != 72:
        raise ValueError(f"teleop.smpl_joints must be (T, 72), got {smpl_joints.shape}.")
    if left_wrist.shape[1] != 3 or right_wrist.shape[1] != 3:
        raise ValueError(
            "teleop.left_wrist_joints and teleop.right_wrist_joints must both be (T, 3); "
            f"got {left_wrist.shape} and {right_wrist.shape}."
        )
    return smpl_joints, left_wrist, right_wrist


def _build_encoder_obs_from_future(
    recorded_encoder_obs: np.ndarray,
    smpl_joints: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    frame: int,
    future_len: int = 10,
) -> tuple[np.ndarray, bool]:
    """Return encoder obs and whether it used reconstructed future fields."""
    if frame + future_len > len(smpl_joints):
        return recorded_encoder_obs[frame].astype(np.float32, copy=True), False

    idx = np.arange(frame, frame + future_len)
    enc_obs = recorded_encoder_obs[frame].astype(np.float32, copy=True)

    s, e = _ENC_OBS_LAYOUT["smpl_joints_10frame_step1"]
    enc_obs[s:e] = smpl_joints[idx].reshape(-1)

    s, e = _ENC_OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"]
    wrist_win = np.concatenate([left_wrist[idx], right_wrist[idx]], axis=1)
    enc_obs[s:e] = wrist_win.reshape(-1)

    return enc_obs, True


def playback(args: argparse.Namespace) -> None:
    (
        states,
        actions,
        _task_indices,
        base_pos,
        base_quat,
        base_ang_vel_rec,
        body_dq_rec,
        enc_obs_rec,
        token_rec,
        _dec_obs_rec,
        dec_action_raw_rec,
        q_target_cmd_rec,
        _smpl_joints_unused,
        _body_root_quat_unused,
    ) = _load_episode(args.dataset_dir, args.episode, sonic=True)

    if actions is None:
        raise ValueError("Future-input playback requires action/action.wbc for hand targets.")
    if enc_obs_rec is None:
        raise ValueError("Dataset must contain sonic.encoder_obs for fallback/template use.")

    teleop_smpl_joints, teleop_left_wrist, teleop_right_wrist = _load_future_fields(
        args.dataset_dir,
        args.episode,
    )

    frame_count = len(states)
    init_frame = args.init_frame
    if init_frame is None:
        init_frame = args.init_substep_frame if args.init_substep_frame is not None else 0
    init_frame = int(np.clip(init_frame, 0, frame_count - 1))
    start_frame = int(np.clip(args.start_frame, 0, frame_count - 1))
    sim_steps = max(1, int(round((1.0 / args.fps) / _SIM_DT)))

    print(f"Episode {args.episode}: {frame_count} frames @ {args.fps:g} Hz")
    print("Encoder future window: 10 continuous frames, current frame + next 9 (step1)")
    print("Encoder fallback: full recorded sonic.encoder_obs when fewer than 10 future frames remain")
    print(f"Init: frame {init_frame} only; Physics: dt={_SIM_DT:g}s, {sim_steps} MuJoCo steps/frame")

    sonic_runner = SonicRunner(args.sonic_encoder, args.sonic_decoder, fps=args.fps, closed_loop=False)
    model = _load_xml(args.env_name)
    model.opt.timestep = _SIM_DT
    data = mujoco.MjData(model)

    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_jid < 0:
        root_jid = None
    body_jids, left_jids, right_jids = _build_joint_indices(model)
    body_ctrl_ids = _build_ctrl_map(model, body_jids)
    left_ctrl_ids = _build_ctrl_map(model, left_jids)
    right_ctrl_ids = _build_ctrl_map(model, right_jids)

    init_base_pos = base_pos[init_frame] if base_pos is not None else None
    init_base_quat = base_quat[init_frame] if base_quat is not None else None
    _set_qpos(
        data,
        model,
        body_jids,
        left_jids,
        right_jids,
        states[init_frame],
        root_jid=root_jid,
        base_pos=init_base_pos,
        base_quat=init_base_quat,
    )
    data.qvel[:] = 0.0
    if body_dq_rec is not None:
        data.qvel[model.jnt_dofadr[body_jids]] = body_dq_rec[init_frame]
    if root_jid is not None:
        root_qvel_adr = model.jnt_dofadr[root_jid]
        if base_pos is not None and frame_count > 1:
            prev_frame = max(init_frame - 1, 0)
            next_frame = min(init_frame + 1, frame_count - 1)
            dt_span = max((next_frame - prev_frame) / args.fps, _SIM_DT)
            data.qvel[root_qvel_adr:root_qvel_adr + 3] = (base_pos[next_frame] - base_pos[prev_frame]) / dt_span
        if base_ang_vel_rec is not None:
            data.qvel[root_qvel_adr + 3:root_qvel_adr + 6] = base_ang_vel_rec[init_frame]
    mujoco.mj_forward(model, data)

    sonic_runner.reset_history(
        states[init_frame],
        init_base_quat if init_base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        actions[init_frame],
        body29_vel0_mujoco=body_dq_rec[init_frame] if body_dq_rec is not None else None,
        base_ang_vel0=base_ang_vel_rec[init_frame] if base_ang_vel_rec is not None else None,
    )

    viewer = None if args.no_viewer else mujoco.viewer.launch_passive(model, data)

    l2_token: list[float] = []
    l2_encoder_obs: list[float] = []
    l2_policy_target_delta: list[float] = []
    l2_q_target: list[float] = []
    l2_state: list[float] = []
    reconstructed_count = 0
    fallback_count = 0

    try:
        for i in range(start_frame, frame_count):
            t0 = time.perf_counter()
            encoder_obs, used_reconstructed_future = _build_encoder_obs_from_future(
                enc_obs_rec,
                teleop_smpl_joints,
                teleop_left_wrist,
                teleop_right_wrist,
                i,
            )
            if used_reconstructed_future:
                reconstructed_count += 1
            else:
                fallback_count += 1

            sim_state = _sim_state43(model, data, body_jids, left_jids, right_jids)
            if root_jid is not None:
                root_qpos_adr = model.jnt_qposadr[root_jid]
                root_qvel_adr = model.jnt_dofadr[root_jid]
                base_quat_for_policy = data.qpos[root_qpos_adr + 3:root_qpos_adr + 7].copy()
                base_ang_vel = data.qvel[root_qvel_adr + 3:root_qvel_adr + 6].copy()
            else:
                base_quat_for_policy = (
                    base_quat[i] if base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                )
                base_ang_vel = None
            body29_vel = data.qvel[model.jnt_dofadr[body_jids]].copy()

            sonic_state43 = sonic_runner.step(
                state43=sim_state.astype(np.float32),
                action43=actions[i],
                base_quat=base_quat_for_policy,
                smpl_joints_win=np.zeros((10, 72), dtype=np.float32),
                body_root_quat_win=np.zeros((10, 4), dtype=np.float32),
                wrist_win=np.zeros((10, 6), dtype=np.float32),
                body29_vel_mujoco=body29_vel,
                base_ang_vel=base_ang_vel,
                encoder_obs_rec=encoder_obs,
            )
            body29_target = np.concatenate([sonic_state43[0:22], sonic_state43[29:36]])

            if args.free_base:
                root_for_step = -1
                bp0 = bp1 = bq0 = bq1 = None
            else:
                root_for_step = root_jid if root_jid is not None else -1
                next_i = min(i + 1, frame_count - 1)
                bp0 = base_pos[i] if base_pos is not None else None
                bp1 = base_pos[next_i] if base_pos is not None else None
                bq0 = base_quat[i] if base_quat is not None else None
                bq1 = base_quat[next_i] if base_quat is not None else None

            _physics_step(
                model,
                data,
                body_ctrl_ids,
                body29_target,
                left_ctrl_ids,
                actions[i][_LEFT_HAND_IDX],
                right_ctrl_ids,
                actions[i][_RIGHT_HAND_IDX],
                n_steps=sim_steps,
                root_jid=root_for_step,
                base_pos_start=bp0,
                base_pos_end=bp1,
                base_quat_start=bq0,
                base_quat_end=bq1,
            )
            mujoco.mj_forward(model, data)

            if args.compare:
                l2_encoder_obs.append(float(np.linalg.norm(encoder_obs - enc_obs_rec[i])))
                if token_rec is not None:
                    token = sonic_runner._enc.run(None, {"obs_dict": encoder_obs.astype(np.float32)[None]})[0][0]
                    l2_token.append(float(np.linalg.norm(token - token_rec[i])))
                if dec_action_raw_rec is not None:
                    q_dev = body29_target - q_target_cmd_rec[i] if q_target_cmd_rec is not None else body29_target
                    l2_policy_target_delta.append(float(np.linalg.norm(q_dev)))
                if q_target_cmd_rec is not None:
                    l2_q_target.append(float(np.linalg.norm(body29_target - q_target_cmd_rec[i])))
                compare_frame = min(i + 1, frame_count - 1)
                sim_state_after = _sim_state43(model, data, body_jids, left_jids, right_jids)
                l2_state.append(float(np.linalg.norm(sim_state_after[_BODY_IDX] - states[compare_frame][_BODY_IDX])))

            if viewer is not None and viewer.is_running():
                viewer.sync()

            if i < start_frame + 10 or (i + 1) % 100 == 0 or i == frame_count - 1:
                msg = f"Frame {i + 1}/{frame_count}  encoder={'future' if used_reconstructed_future else 'recorded'}"
                if args.compare:
                    if l2_encoder_obs:
                        msg += f"  L2_enc_obs={l2_encoder_obs[-1]:.3e}"
                    if l2_token:
                        msg += f"  L2_token={l2_token[-1]:.3e}"
                    if l2_q_target:
                        msg += f"  L2_q_target={l2_q_target[-1]:.3e}"
                    if l2_state:
                        msg += f"  L2_state={l2_state[-1]:.3e}"
                print(msg)

            sleep_s = (1.0 / args.fps) - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        if viewer is not None:
            viewer.close()

    print(f"Encoder source counts: reconstructed_future={reconstructed_count}, recorded_fallback={fallback_count}")
    if args.compare:
        for name, values in (
            ("encoder_obs", l2_encoder_obs),
            ("token", l2_token),
            ("policy_target_delta", l2_policy_target_delta),
            ("q_target", l2_q_target),
            ("sim_body_state", l2_state),
        ):
            if values:
                arr = np.asarray(values)
                print(f"{name}: mean={arr.mean():.3e} median={np.median(arr):.3e} max={arr.max():.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--env_name", default="default")
    parser.add_argument("--sonic_encoder", required=True)
    parser.add_argument("--sonic_decoder", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--free_base", action="store_true")
    parser.add_argument("--init_frame", type=int, default=None)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--no_viewer", action="store_true")

    # Kept for command compatibility; these only choose the init frame.
    parser.add_argument("--init_substep_frame", type=int, default=None)
    parser.add_argument("--init_substep_index", type=int, default=0)
    parser.add_argument("--init_substep_all_inputs", action="store_true")

    playback(parser.parse_args())


if __name__ == "__main__":
    main()
