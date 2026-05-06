"""Playback recorded SONIC ONNX inputs through MuJoCo PD physics.

This script intentionally does not replay recorded MuJoCo substeps.  It uses one
dataset frame to initialize the environment, then runs the ONNX encoder/decoder
from the recorded input tensors and lets MuJoCo physics advance from there.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from playback_lerobot import (
    _BODY_IDX,
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


def _sim_state43(model: mujoco.MjModel, data: mujoco.MjData, body_jids, left_jids, right_jids) -> np.ndarray:
    state = np.zeros(43, dtype=np.float64)
    body29 = data.qpos[model.jnt_qposadr[body_jids]]
    state[0:22] = body29[0:22]
    state[29:36] = body29[22:29]
    if len(left_jids):
        state[22:29] = data.qpos[model.jnt_qposadr[left_jids]]
    if len(right_jids):
        state[36:43] = data.qpos[model.jnt_qposadr[right_jids]]
    return state


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
        dec_obs_rec,
        dec_action_raw_rec,
        q_target_cmd_rec,
        _smpl_joints,
        _body_root_quat,
    ) = _load_episode(args.dataset_dir, args.episode, sonic=True)

    if actions is None:
        raise ValueError("ONNX-input playback requires action/action.wbc for hand targets.")
    if enc_obs_rec is None or dec_obs_rec is None:
        raise ValueError("Dataset must contain sonic.encoder_obs and sonic.decoder_obs.")

    frame_count = len(states)
    init_frame = args.init_frame
    if init_frame is None:
        init_frame = args.init_substep_frame if args.init_substep_frame is not None else 0
    init_frame = int(np.clip(init_frame, 0, frame_count - 1))
    start_frame = int(np.clip(args.start_frame, 0, frame_count - 1))

    sim_steps = max(1, int(round((1.0 / args.fps) / _SIM_DT)))
    print(f"Episode {args.episode}: {frame_count} frames @ {args.fps:g} Hz")
    print(f"Init: frame {init_frame} only; ignoring recorded substep fields")
    print(f"Physics: dt={_SIM_DT:g}s, {sim_steps} MuJoCo steps/frame, free_base={args.free_base}")

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

    init_base_quat_for_policy = (
        init_base_quat if init_base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    sonic_runner.reset_history(
        states[init_frame],
        init_base_quat_for_policy,
        actions[init_frame],
        body29_vel0_mujoco=body_dq_rec[init_frame] if body_dq_rec is not None else None,
        base_ang_vel0=base_ang_vel_rec[init_frame] if base_ang_vel_rec is not None else None,
    )

    viewer = None if args.no_viewer else mujoco.viewer.launch_passive(model, data)
    renderer = video_writer = cv2 = None
    if args.output_video:
        import cv2 as _cv2

        model.vis.global_.offwidth = args.video_width
        model.vis.global_.offheight = args.video_height
        renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = _cv2.VideoWriter(
            args.output_video,
            fourcc,
            args.fps,
            (args.video_width, args.video_height),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {args.output_video}")
        cv2 = _cv2
        print(f"Saving video -> {args.output_video} ({args.video_width}x{args.video_height} @ {args.fps:g} fps)")

    l2_token: list[float] = []
    l2_policy_target_delta: list[float] = []
    l2_q_target: list[float] = []
    l2_state: list[float] = []

    try:
        for i in range(start_frame, frame_count):
            t0 = time.perf_counter()
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
                encoder_obs_rec=enc_obs_rec[i],
            )
            body29_target = np.concatenate([sonic_state43[0:22], sonic_state43[29:36]])

            left_target = actions[i][_LEFT_HAND_IDX]
            right_target = actions[i][_RIGHT_HAND_IDX]

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
                left_target,
                right_ctrl_ids,
                right_target,
                n_steps=sim_steps,
                root_jid=root_for_step,
                base_pos_start=bp0,
                base_pos_end=bp1,
                base_quat_start=bq0,
                base_quat_end=bq1,
            )
            mujoco.mj_forward(model, data)

            if args.compare:
                if token_rec is not None:
                    token = sonic_runner._enc.run(None, {"obs_dict": enc_obs_rec[i].astype(np.float32)[None]})[0][0]
                    l2_token.append(float(np.linalg.norm(token - token_rec[i])))
                if dec_action_raw_rec is not None:
                    q_dev = body29_target - q_target_cmd_rec[i] if q_target_cmd_rec is not None else body29_target
                    l2_policy_target_delta.append(float(np.linalg.norm(q_dev)))
                if q_target_cmd_rec is not None:
                    l2_q_target.append(float(np.linalg.norm(body29_target - q_target_cmd_rec[i])))
                sim_state = _sim_state43(model, data, body_jids, left_jids, right_jids)
                compare_frame = min(i + 1, frame_count - 1)
                l2_state.append(float(np.linalg.norm(sim_state[_BODY_IDX] - states[compare_frame][_BODY_IDX])))

            if renderer is not None and video_writer is not None and cv2 is not None:
                renderer.update_scene(data, camera=args.camera)
                rgb = renderer.render()
                video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if viewer is not None and viewer.is_running():
                viewer.sync()

            if i < start_frame + 10 or (i + 1) % 100 == 0 or i == frame_count - 1:
                msg = f"Frame {i + 1}/{frame_count}"
                if args.compare:
                    if l2_token:
                        msg += f"  L2_token={l2_token[-1]:.3e}"
                    if l2_policy_target_delta:
                        msg += f"  L2_target_delta={l2_policy_target_delta[-1]:.3e}"
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
        if renderer is not None:
            renderer.close()
        if video_writer is not None:
            video_writer.release()

    if args.compare:
        for name, values in (
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
    parser.add_argument("--output_video", default=None)
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=360)

    # Backward-compatible no-op flags from the older substep experiment.
    parser.add_argument("--init_substep_frame", type=int, default=None)
    parser.add_argument("--init_substep_index", type=int, default=0)
    parser.add_argument("--init_substep_all_inputs", action="store_true")

    playback(parser.parse_args())


if __name__ == "__main__":
    main()
