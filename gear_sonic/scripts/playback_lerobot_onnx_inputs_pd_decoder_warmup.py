"""Hybrid ONNX-input playback with recorded warmup, then closed-loop rollout."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq

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
from playback_lerobot_onnx_inputs_pd import _sim_state43

_SUBSTEP_TIME_EPS = 1e-12


def _episode_path(dataset_dir: str, episode: int) -> str:
    chunk = episode // 1000
    return f"{dataset_dir}/data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"


def _load_optional_vector(table, name: str) -> np.ndarray:
    arr = np.array([r.as_py() for r in table.column(name)], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != table.num_rows:
        raise ValueError(f"{name} must have shape (T, N); got {arr.shape}")
    return arr


def _load_optional_matrix(dataset_dir: str, episode: int, names: list[str]) -> dict[str, np.ndarray]:
    path = _episode_path(dataset_dir, episode)
    schema_names = set(pq.read_schema(path).names)
    present = [name for name in names if name in schema_names]
    if not present:
        return {}
    table = pq.read_table(path, columns=present)
    return {name: _load_optional_vector(table, name) for name in present}


def _apply_recorded_substep_hand_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    recorded: dict[str, np.ndarray] | None,
    left_jids: np.ndarray,
    right_jids: np.ndarray,
    frame: int,
) -> bool:
    if recorded is None or "qpos" not in recorded or "qvel" not in recorded or "time" not in recorded:
        return False
    if frame < 0 or frame >= recorded["qpos"].shape[0]:
        return False
    valid = np.where(np.isfinite(recorded["time"][frame]))[0]
    if len(valid) == 0:
        return False
    substep = int(valid[0])
    qpos = recorded["qpos"][frame, substep]
    qvel = recorded["qvel"][frame, substep]
    if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
        return False

    data.qpos[model.jnt_qposadr[left_jids]] = qpos[model.jnt_qposadr[left_jids]]
    data.qvel[model.jnt_dofadr[left_jids]] = qvel[model.jnt_dofadr[left_jids]]
    data.qpos[model.jnt_qposadr[right_jids]] = qpos[model.jnt_qposadr[right_jids]]
    data.qvel[model.jnt_dofadr[right_jids]] = qvel[model.jnt_dofadr[right_jids]]
    return True


def _load_recorded_substeps(dataset_dir: str, episode: int, model: mujoco.MjModel) -> dict[str, np.ndarray]:
    path = _episode_path(dataset_dir, episode)
    required = [
        "robot.motor_pd_substep_sim_time",
        "robot.mujoco_substep_qpos",
        "robot.mujoco_substep_qvel",
        "robot.mujoco_substep_ctrl",
        "robot.mujoco_substep_qfrc_applied",
        "robot.mujoco_substep_xfrc_applied",
        "robot.mujoco_substep_qacc_warmstart",
    ]
    names = pq.read_schema(path).names
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing recorded substep columns: {missing}")
    optional = ["robot.motor_pd_substep_q_des"]
    table = pq.read_table(path, columns=required + [name for name in optional if name in names])

    def reshape(name: str, width: int) -> np.ndarray:
        arr = _load_optional_vector(table, name)
        if arr.shape[1] % width != 0:
            raise ValueError(f"{name} width {arr.shape[1]} is not divisible by {width}")
        return arr.reshape(arr.shape[0], arr.shape[1] // width, width)

    xfrc = reshape("robot.mujoco_substep_xfrc_applied", model.nbody * 6)
    recorded = {
        "time": _load_optional_vector(table, "robot.motor_pd_substep_sim_time"),
        "qpos": reshape("robot.mujoco_substep_qpos", model.nq),
        "qvel": reshape("robot.mujoco_substep_qvel", model.nv),
        "ctrl": reshape("robot.mujoco_substep_ctrl", model.nu),
        "qfrc": reshape("robot.mujoco_substep_qfrc_applied", model.nv),
        "xfrc": xfrc.reshape(xfrc.shape[0], xfrc.shape[1], model.nbody, 6),
        "warm": reshape("robot.mujoco_substep_qacc_warmstart", model.nv),
    }
    if "robot.motor_pd_substep_q_des" in table.column_names:
        recorded["body_q_des"] = reshape("robot.motor_pd_substep_q_des", 29)
    return recorded


def _recorded_body_q_des(recorded: dict[str, np.ndarray] | None, frame: int) -> np.ndarray | None:
    if recorded is None or "body_q_des" not in recorded:
        return None
    frame_q_des = recorded["body_q_des"][frame]
    valid = np.where(np.isfinite(frame_q_des).all(axis=1))[0]
    if len(valid) == 0:
        return None
    return frame_q_des[int(valid[0])].copy()


def _apply_recorded_substep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    recorded: dict[str, np.ndarray],
    frame: int,
    substep: int,
    *,
    step: bool,
) -> bool:
    substep_time = float(recorded["time"][frame, substep])
    qpos = recorded["qpos"][frame, substep]
    qvel = recorded["qvel"][frame, substep]
    ctrl = recorded["ctrl"][frame, substep]
    qfrc = recorded["qfrc"][frame, substep]
    xfrc = recorded["xfrc"][frame, substep]
    warm = recorded["warm"][frame, substep]
    if not (
        np.isfinite(substep_time)
        and np.isfinite(qpos).all()
        and np.isfinite(qvel).all()
        and np.isfinite(ctrl).all()
        and np.isfinite(qfrc).all()
        and np.isfinite(xfrc).all()
        and np.isfinite(warm).all()
    ):
        return False
    if step and substep_time < data.time - _SUBSTEP_TIME_EPS:
        return False
    data.time = substep_time
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    data.qfrc_applied[:] = qfrc
    data.xfrc_applied[:] = xfrc
    data.qacc_warmstart[:] = warm
    if step:
        mujoco.mj_step(model, data)
    else:
        mujoco.mj_forward(model, data)
    return True


def _physics_step_recorded_hand_gains(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ctrl_ids: np.ndarray,
    body29_target: np.ndarray,
    left_ctrl_ids: np.ndarray,
    left7_target: np.ndarray,
    right_ctrl_ids: np.ndarray,
    right7_target: np.ndarray,
    *,
    hand_cmds: dict[str, np.ndarray],
    frame: int,
    n_steps: int,
    root_jid: int = -1,
    base_pos_start: np.ndarray | None = None,
    base_pos_end: np.ndarray | None = None,
    base_quat_start: np.ndarray | None = None,
    base_quat_end: np.ndarray | None = None,
) -> None:
    """Body PD with recorded Dex3 hand kp/kd/targets when available."""
    from playback_lerobot import (
        _KP_BODY29,
        _KD_BODY29,
        _KP_HAND,
        _KD_HAND,
        _SIM_DT,
        _TORQUE_LIMIT_BODY29,
        _TORQUE_LIMIT_HAND,
        _quat_conj,
        _quat_mult,
        _quat_slerp,
        _quat_to_rot_matrix,
    )

    def hand_values(side: str, fallback_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = hand_cmds.get(f"teleop.{side}_hand_joints")
        kp = hand_cmds.get(f"robot.{side}_hand_motor_kp")
        kd = hand_cmds.get(f"robot.{side}_hand_motor_kd")
        target_arr = fallback_target if target is None else target[frame]
        kp_arr = np.full(7, _KP_HAND, dtype=np.float64) if kp is None else kp[frame]
        kd_arr = np.full(7, _KD_HAND, dtype=np.float64) if kd is None else kd[frame]
        return target_arr, kp_arr, kd_arr

    left_target, left_kp, left_kd = hand_values("left", left7_target)
    right_target, right_kp, right_kd = hand_values("right", right7_target)

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
        lin_vel = (base_pos_end - base_pos_start) / policy_dt
        q_diff = _quat_mult(_quat_conj(base_quat_start), base_quat_end)
        half_angle = np.arccos(np.clip(abs(float(q_diff[0])), 0.0, 1.0))
        if half_angle >= 1e-8:
            axis_body = q_diff[1:4] / np.sin(half_angle)
            ang_vel_world = _quat_to_rot_matrix(base_quat_start) @ (
                axis_body * (2.0 * half_angle / policy_dt)
            )

    for k in range(n_steps):
        if drive_base:
            frac = k / n_steps
            data.qpos[qpos_adr:qpos_adr + 3] = (1.0 - frac) * base_pos_start + frac * base_pos_end
            data.qpos[qpos_adr + 3:qpos_adr + 7] = _quat_slerp(base_quat_start, base_quat_end, frac)
            data.qvel[qvel_adr:qvel_adr + 3] = lin_vel
            data.qvel[qvel_adr + 3:qvel_adr + 6] = ang_vel_world

        ctrl = np.zeros(model.nu, dtype=np.float64)

        for j, (cid, kp, kd, tlim) in enumerate(
            zip(body_ctrl_ids, _KP_BODY29, _KD_BODY29, _TORQUE_LIMIT_BODY29)
        ):
            if cid < 0:
                continue
            jid = model.actuator(cid).trnid[0]
            q = data.qpos[model.jnt_qposadr[jid]]
            dq = data.qvel[model.jnt_dofadr[jid]]
            tau = kp * (body29_target[j] - q) + kd * (0.0 - dq)
            ctrl[cid] = np.clip(tau, -tlim, tlim)

        for target_arr, kp_arr, kd_arr, ctrl_ids in (
            (left_target, left_kp, left_kd, left_ctrl_ids),
            (right_target, right_kp, right_kd, right_ctrl_ids),
        ):
            for j, cid in enumerate(ctrl_ids):
                if cid < 0:
                    continue
                jid = model.actuator(cid).trnid[0]
                q = data.qpos[model.jnt_qposadr[jid]]
                dq = data.qvel[model.jnt_dofadr[jid]]
                tau = kp_arr[j] * (target_arr[j] - q) + kd_arr[j] * (0.0 - dq)
                ctrl[cid] = np.clip(tau, -_TORQUE_LIMIT_HAND, _TORQUE_LIMIT_HAND)

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)


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
        raise ValueError("ONNX warmup playback requires action/action.wbc.")
    if enc_obs_rec is None or dec_obs_rec is None:
        raise ValueError("Dataset must contain sonic.encoder_obs and sonic.decoder_obs.")

    frame_count = len(states)
    init_frame = args.init_frame
    if init_frame is None:
        init_frame = args.init_substep_frame if args.init_substep_frame is not None else 0
    init_frame = int(np.clip(init_frame, 0, frame_count - 1))
    start_frame = int(np.clip(args.start_frame, 0, frame_count - 1))
    decoder_warmup_frames = max(0, int(args.recorded_decoder_warmup_frames))
    physics_warmup_frames = max(0, int(args.recorded_physics_warmup_frames))
    sim_steps = max(1, int(round((1.0 / args.fps) / _SIM_DT)))

    print(f"Episode {args.episode}: {frame_count} frames @ {args.fps:g} Hz")
    print(f"Init: frame {init_frame} only; warmup logic is local to this script")
    print(f"Physics: dt={_SIM_DT:g}s, {sim_steps} MuJoCo steps/frame, free_base={args.free_base}")
    print(f"Decoder obs warmup: {decoder_warmup_frames} frames")
    print(f"Recorded physics warmup: {physics_warmup_frames} frames")
    print(f"LowCmd target latency emulation: {not args.no_lowcmd_latency}")
    print(f"Recorded body dq history: {args.recorded_body_dq_history}")

    sonic_runner = SonicRunner(args.sonic_encoder, args.sonic_decoder, fps=args.fps, closed_loop=False)
    model = _load_xml(args.env_name)
    model.opt.timestep = _SIM_DT
    data = mujoco.MjData(model)
    recorded_substeps = _load_recorded_substeps(args.dataset_dir, args.episode, model) if physics_warmup_frames else None
    recorded_hand_cmds = _load_optional_matrix(
        args.dataset_dir,
        args.episode,
        [
            "teleop.left_hand_joints",
            "teleop.right_hand_joints",
            "robot.left_hand_motor_kp",
            "robot.left_hand_motor_kd",
            "robot.right_hand_motor_kp",
            "robot.right_hand_motor_kd",
        ],
    )
    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_jid < 0:
        root_jid = None
    body_jids, left_jids, right_jids = _build_joint_indices(model)
    body_ctrl_ids = _build_ctrl_map(model, body_jids)
    left_ctrl_ids = _build_ctrl_map(model, left_jids)
    right_ctrl_ids = _build_ctrl_map(model, right_jids)

    init_base_pos = base_pos[init_frame] if base_pos is not None else None
    init_base_quat = base_quat[init_frame] if base_quat is not None else None
    if recorded_substeps is not None:
        init_substep_frame = int(np.clip(args.init_substep_frame or 0, 0, frame_count - 1))
        max_substeps = min(v.shape[1] for v in recorded_substeps.values())
        init_substep_index = int(np.clip(args.init_substep_index, 0, max_substeps - 1))
        if not _apply_recorded_substep(
            model,
            data,
            recorded_substeps,
            init_substep_frame,
            init_substep_index,
            step=False,
        ):
            raise ValueError(f"Could not initialize from recorded substep f{init_substep_frame}s{init_substep_index}.")
        if root_jid is not None:
            root_qpos_adr = model.jnt_qposadr[root_jid]
            init_base_quat = data.qpos[root_qpos_adr + 3:root_qpos_adr + 7].copy()
        print(
            f"Initial state: recorded substep f{init_substep_frame}s{init_substep_index} "
            "(all recorded MuJoCo inputs)"
        )
    else:
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
        if root_jid is not None and base_ang_vel_rec is not None:
            root_qvel_adr = model.jnt_dofadr[root_jid]
            data.qvel[root_qvel_adr + 3:root_qvel_adr + 6] = base_ang_vel_rec[init_frame]
        mujoco.mj_forward(model, data)

    sonic_runner.reset_history(
        states[init_frame],
        init_base_quat if init_base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]),
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
    l2_q_target: list[float] = []
    l2_physics_target: list[float] = []
    l2_state: list[float] = []
    pending_body29_target: np.ndarray | None = None
    extra_compare_frames = {12, 13}

    try:
        for i in range(start_frame, frame_count):
            t0 = time.perf_counter()

            if (
                args.reset_handoff_substep
                and recorded_substeps is not None
                and i == start_frame + physics_warmup_frames
            ):
                valid = np.where(np.isfinite(recorded_substeps["time"][i]))[0]
                if len(valid):
                    _apply_recorded_substep(model, data, recorded_substeps, i, int(valid[0]), step=False)

            if i >= start_frame + physics_warmup_frames:
                _apply_recorded_substep_hand_state(
                    model,
                    data,
                    recorded_substeps,
                    left_jids,
                    right_jids,
                    i,
                )

            sim_state = _sim_state43(model, data, body_jids, left_jids, right_jids)
            if root_jid is not None:
                root_qpos_adr = model.jnt_qposadr[root_jid]
                root_qvel_adr = model.jnt_dofadr[root_jid]
                base_quat_for_policy = data.qpos[root_qpos_adr + 3:root_qpos_adr + 7].copy()
                base_ang_vel = data.qvel[root_qvel_adr + 3:root_qvel_adr + 6].copy()
            else:
                base_quat_for_policy = base_quat[i] if base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
                base_ang_vel = None
            body29_vel = data.qvel[model.jnt_dofadr[body_jids]].copy()
            state43_for_decoder = sim_state
            base_quat_for_decoder = base_quat_for_policy
            base_ang_vel_for_decoder = base_ang_vel
            body29_vel_for_decoder = body29_vel
            if args.recorded_body_dq_history and body_dq_rec is not None:
                body29_vel_for_decoder = body_dq_rec[i]
            if args.recorded_lowstate_history:
                state43_for_decoder = states[i].astype(np.float64)
                if base_quat is not None:
                    base_quat_for_decoder = base_quat[i]
                if base_ang_vel_rec is not None:
                    base_ang_vel_for_decoder = base_ang_vel_rec[i]
                if body_dq_rec is not None:
                    body29_vel_for_decoder = body_dq_rec[i]
            decoder_history_obs = (
                dec_obs_rec[i]
                if i == start_frame + decoder_warmup_frames
                else None
            )

            sonic_state43 = sonic_runner.step(
                state43=state43_for_decoder.astype(np.float32),
                action43=actions[i],
                base_quat=base_quat_for_decoder,
                smpl_joints_win=np.zeros((10, 72), dtype=np.float32),
                body_root_quat_win=np.zeros((10, 4), dtype=np.float32),
                wrist_win=np.zeros((10, 6), dtype=np.float32),
                body29_vel_mujoco=body29_vel_for_decoder,
                base_ang_vel=base_ang_vel_for_decoder,
                encoder_obs_rec=enc_obs_rec[i],
                decoder_obs_rec=dec_obs_rec[i] if i < start_frame + decoder_warmup_frames else None,
                decoder_history_obs_rec=decoder_history_obs,
                history_action_raw_rec=(
                    dec_action_raw_rec[i] if args.recorded_last_action_history else None
                ),
            )
            body29_target = np.concatenate([sonic_state43[0:22], sonic_state43[29:36]])
            physics_body29_target = body29_target
            handoff_frame = start_frame + physics_warmup_frames
            if not args.no_lowcmd_latency:
                if i == handoff_frame:
                    # The simulator has already received an older LowCmd at the
                    # first live step. Seed from the recorded q_des so the
                    # handoff uses the same target timing as the teleop run.
                    physics_body29_target = _recorded_body_q_des(recorded_substeps, i)
                    if physics_body29_target is None:
                        physics_body29_target = pending_body29_target
                    if physics_body29_target is None:
                        physics_body29_target = body29_target
                elif pending_body29_target is not None:
                    physics_body29_target = pending_body29_target
                pending_body29_target = body29_target.copy()

            if recorded_substeps is not None and i < start_frame + physics_warmup_frames:
                max_substeps = min(v.shape[1] for v in recorded_substeps.values())
                substeps_to_apply = max_substeps
                is_final_warmup_frame = i == start_frame + physics_warmup_frames - 1
                if is_final_warmup_frame and i + 1 < frame_count and not args.step_final_warmup_overlap:
                    current_times = recorded_substeps["time"][i]
                    next_times = recorded_substeps["time"][i + 1]
                    valid_current = np.where(np.isfinite(current_times))[0]
                    valid_next = np.where(np.isfinite(next_times))[0]
                    if len(valid_current) and len(valid_next):
                        last_substep = int(valid_current[-1])
                        next_first_substep = int(valid_next[0])
                        if abs(current_times[last_substep] - next_times[next_first_substep]) <= _SUBSTEP_TIME_EPS:
                            # Leave MuJoCo at the overlapping pre-step state so
                            # the first live frame can drive it. Stepping it here
                            # would advance one substep too far before handoff.
                            substeps_to_apply = min(substeps_to_apply, last_substep)
                for substep in range(substeps_to_apply):
                    _apply_recorded_substep(model, data, recorded_substeps, i, substep, step=True)
                mujoco.mj_forward(model, data)
            else:
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
                if recorded_hand_cmds:
                    _physics_step_recorded_hand_gains(
                        model,
                        data,
                        body_ctrl_ids,
                        physics_body29_target,
                        left_ctrl_ids,
                        actions[i][_LEFT_HAND_IDX],
                        right_ctrl_ids,
                        actions[i][_RIGHT_HAND_IDX],
                        hand_cmds=recorded_hand_cmds,
                        frame=i,
                        n_steps=sim_steps,
                        root_jid=root_for_step,
                        base_pos_start=bp0,
                        base_pos_end=bp1,
                        base_quat_start=bq0,
                        base_quat_end=bq1,
                    )
                else:
                    _physics_step(
                        model,
                        data,
                        body_ctrl_ids,
                        physics_body29_target,
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
                _apply_recorded_substep_hand_state(
                    model,
                    data,
                    recorded_substeps,
                    left_jids,
                    right_jids,
                    min(i + 1, frame_count - 1),
                )
                mujoco.mj_forward(model, data)

            if args.compare:
                if token_rec is not None:
                    token = sonic_runner._enc.run(None, {"obs_dict": enc_obs_rec[i].astype(np.float32)[None]})[0][0]
                    l2_token.append(float(np.linalg.norm(token - token_rec[i])))
                if q_target_cmd_rec is not None:
                    l2_q_target.append(float(np.linalg.norm(body29_target - q_target_cmd_rec[i])))
                recorded_q_des = _recorded_body_q_des(recorded_substeps, i)
                if recorded_q_des is not None:
                    l2_physics_target.append(float(np.linalg.norm(physics_body29_target - recorded_q_des)))
                compare_frame = min(i + 1, frame_count - 1)
                sim_state = _sim_state43(model, data, body_jids, left_jids, right_jids)
                l2_state.append(float(np.linalg.norm(sim_state[_BODY_IDX] - states[compare_frame][_BODY_IDX])))

            if renderer is not None and video_writer is not None and cv2 is not None:
                renderer.update_scene(data, camera=args.camera)
                rgb = renderer.render()
                video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if viewer is not None and viewer.is_running():
                viewer.sync()
            frame_no = i + 1
            if i < start_frame + 10 or frame_no in extra_compare_frames or frame_no % 100 == 0 or i == frame_count - 1:
                msg = f"Frame {frame_no}/{frame_count}"
                if args.compare:
                    if l2_token:
                        msg += f"  L2_token={l2_token[-1]:.3e}"
                    if l2_q_target:
                        msg += f"  L2_q_target={l2_q_target[-1]:.3e}"
                    if l2_physics_target:
                        msg += f"  L2_physics_target={l2_physics_target[-1]:.3e}"
                    if l2_state:
                        msg += f"  L2_state={l2_state[-1]:.3e}"
                print(msg)
            sleep_s = (1.0 / args.fps) - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        close_runner = getattr(sonic_runner, "close", None)
        if close_runner is not None:
            close_runner()
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
        if video_writer is not None:
            video_writer.release()

    if args.compare:
        for name, values in (
            ("token", l2_token),
            ("q_target", l2_q_target),
            ("physics_target", l2_physics_target),
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
    parser.add_argument(
        "--recorded_decoder_warmup_frames",
        type=int,
        default=10,
        help="Number of playback frames that use recorded sonic.decoder_obs before closed-loop rollout.",
    )
    parser.add_argument(
        "--recorded_physics_warmup_frames",
        type=int,
        default=10,
        help="Number of playback frames that replay recorded MuJoCo substeps before closed-loop rollout.",
    )
    parser.add_argument(
        "--no_lowcmd_latency",
        action="store_true",
        help=(
            "Apply the freshly inferred q_target immediately. By default, live "
            "physics uses one-frame LowCmd latency and seeds the handoff target "
            "from recorded robot.motor_pd_substep_q_des when available."
        ),
    )
    parser.add_argument(
        "--recorded_lowstate_history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build live decoder history from recorded LowState-style fields "
            "(observation.state, robot.body_dq, robot.base_ang_vel) instead of "
            "MuJoCo internal qpos/qvel. This matches how C++ teleop fills "
            "StateLogger history for decoder-input comparison, but it is not a "
            "true closed-loop MuJoCo rollout."
        ),
    )
    parser.add_argument(
        "--recorded_body_dq_history",
        action="store_true",
        help=(
            "Use recorded robot.body_dq for the decoder his_body_joint_velocities "
            "history while leaving body q/base history sourced from the live "
            "playback state. This matches the teleop decoder velocity source "
            "without fully switching to recorded_lowstate_history."
        ),
    )
    parser.add_argument(
        "--recorded_last_action_history",
        action="store_true",
        help=(
            "For decoder diagnostics, store recorded sonic.decoder_action_raw "
            "into the next step's his_last_actions history while still running "
            "the current Python decoder. This mirrors C++ teleop's previous "
            "TensorRT action history."
        ),
    )
    parser.add_argument(
        "--reset_handoff_substep",
        action="store_true",
        help=(
            "At the first live physics frame after recorded warmup, reset MuJoCo "
            "to that frame's recorded first substep. By default the playback now "
            "carries the state produced by the previous warmup frame and lets "
            "Frame 12 drive forward from Frame 11."
        ),
    )
    parser.add_argument(
        "--step_final_warmup_overlap",
        action="store_true",
        help=(
            "During the last recorded warmup frame, also step the final substep "
            "when it overlaps the next frame's first substep. By default this "
            "overlapping substep is left unstepped so the first live frame starts "
            "from that pre-step state."
        ),
    )

    # Kept so the old command shape still works; these only choose the init frame.
    parser.add_argument("--init_substep_frame", type=int, default=None)
    parser.add_argument("--init_substep_index", type=int, default=0)
    parser.add_argument("--init_substep_all_inputs", action="store_true")

    playback(parser.parse_args())


if __name__ == "__main__":
    main()
