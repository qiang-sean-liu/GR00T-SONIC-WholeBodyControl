"""Physics replay for LeRobot datasets.

Default mode replays recorded ``observation.state`` through MuJoCo physics by
tracking each frame with joint-space PD control. If ``--sonic_encoder`` and
``--sonic_decoder`` are provided, this script delegates to
``playback_lerobot.py`` in SONIC physics mode so the same policy replay path is
used from this entrypoint.

When ``robot.base_pos`` / ``robot.base_quat`` are present, state-only mode uses
the recorded root pose only to initialize the robot at the beginning. After
that, the robot moves freely under physics while the recorded joint states are
tracked with PD control. If the dataset does not contain usable base pose, it
falls back to a standing initialization pose.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from gear_sonic.scripts.playback_lerobot import (
    _BODY_IDX,
    _KD_BODY29,
    _KD_HAND,
    _KP_BODY29,
    _KP_HAND,
    _LEFT_HAND_IDX,
    _RIGHT_HAND_IDX,
    _ROOT_HEIGHT,
    _SIM_DT,
    _SIM_STEPS_PER_POLICY,
    _TORQUE_LIMIT_BODY29,
    _TORQUE_LIMIT_HAND,
    SonicRunner,
    _build_ctrl_map,
    _build_joint_indices,
    _load_episode,
    _load_xml,
    playback,
    _physics_step,
    _set_qpos,
)


def _set_root_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_jid: int | None,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> None:
    if root_jid is None:
        return

    qpos_adr = model.jnt_qposadr[root_jid]
    qvel_adr = model.jnt_dofadr[root_jid]
    data.qpos[qpos_adr : qpos_adr + 3] = base_pos
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = base_quat
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def _physics_track_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_jids: np.ndarray,
    left_jids: np.ndarray,
    right_jids: np.ndarray,
    body_ctrl_ids: np.ndarray,
    left_ctrl_ids: np.ndarray,
    right_ctrl_ids: np.ndarray,
    target_state43: np.ndarray,
    n_steps: int,
) -> None:
    body29_target = target_state43[_BODY_IDX]
    left7_target = target_state43[_LEFT_HAND_IDX]
    right7_target = target_state43[_RIGHT_HAND_IDX]

    for _ in range(n_steps):
        ctrl = np.zeros(model.nu)

        for j, (cid, kp, kd, tlim) in enumerate(
            zip(body_ctrl_ids, _KP_BODY29, _KD_BODY29, _TORQUE_LIMIT_BODY29)
        ):
            if cid < 0:
                continue
            q = data.qpos[model.jnt_qposadr[model.actuator(cid).trnid[0]]]
            dq = data.qvel[model.jnt_dofadr[model.actuator(cid).trnid[0]]]
            tau = kp * (body29_target[j] - q) + kd * (0.0 - dq)
            ctrl[cid] = np.clip(tau, -tlim, tlim)

        for target_arr, ctrl_ids in ((left7_target, left_ctrl_ids), (right7_target, right_ctrl_ids)):
            for j, cid in enumerate(ctrl_ids):
                if cid < 0:
                    continue
                q = data.qpos[model.jnt_qposadr[model.actuator(cid).trnid[0]]]
                dq = data.qvel[model.jnt_dofadr[model.actuator(cid).trnid[0]]]
                tau = _KP_HAND * (target_arr[j] - q) + _KD_HAND * (0.0 - dq)
                ctrl[cid] = np.clip(tau, -_TORQUE_LIMIT_HAND, _TORQUE_LIMIT_HAND)

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)


def playback_physics_state(
    dataset_dir: str,
    episode: int,
    env_name: str,
    fps: float,
    no_viewer: bool,
    sim_substeps: int,
    drive_recorded_base: bool,
) -> None:
    states, _, task_indices, base_pos, base_quat, *_ = _load_episode(dataset_dir, episode, sonic=False)
    total_frames = len(states)
    task_index = task_indices[0] if task_indices else None

    model = _load_xml(env_name)
    model.opt.timestep = _SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    root_jid_list = [i for i in range(model.njnt) if model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE]
    root_jid = root_jid_list[0] if root_jid_list else None
    fixed_base_pos = np.array([0.0, 0.0, _ROOT_HEIGHT], dtype=np.float64)
    fixed_base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    base_pos_range = None
    if base_pos is not None:
        base_pos_range = np.max(base_pos, axis=0) - np.min(base_pos, axis=0)
        # Treat nearly-constant base position as unusable fallback data rather than a real root trajectory.
        if float(np.max(base_pos_range)) < 1e-6:
            base_pos = None

    body_jids, left_jids, right_jids = _build_joint_indices(model)
    body_ctrl_ids = _build_ctrl_map(model, body_jids)
    left_ctrl_ids = _build_ctrl_map(model, left_jids)
    right_ctrl_ids = _build_ctrl_map(model, right_jids)

    use_recorded_base = base_pos is not None and base_quat is not None
    base_pos_arr = base_pos if use_recorded_base else None
    base_quat_arr = base_quat if use_recorded_base else None
    initial_base_pos = (
        np.asarray(base_pos_arr[0], dtype=np.float64)
        if base_pos_arr is not None
        else fixed_base_pos
    )
    initial_base_quat = (
        np.asarray(base_quat_arr[0], dtype=np.float64)
        if base_quat_arr is not None
        else fixed_base_quat
    )

    _set_qpos(
        data,
        model,
        body_jids,
        left_jids,
        right_jids,
        np.asarray(states[0], dtype=np.float64),
        root_jid=root_jid,
        base_pos=initial_base_pos,
        base_quat=initial_base_quat,
    )
    mujoco.mj_forward(model, data)

    print(f"Episode {episode}: {total_frames} frames @ {fps:.0f} Hz")
    print(f"Task index: {task_index}")
    print(f"Environment: {env_name}")
    if use_recorded_base:
        if drive_recorded_base:
            print(
                "Mode: physics replay of recorded observation.state with recorded base pose "
                f"driven every step (base_pos range xyz in dataset = {base_pos_range})"
            )
        else:
            print(
                "Mode: physics replay of recorded observation.state with recorded initial floating "
                f"base pose (base_pos range xyz in dataset = {base_pos_range})"
            )
    elif base_quat is not None and base_pos_range is not None:
        print(
            "Mode: physics replay of recorded observation.state with recorded initial base "
            "orientation only (base_pos was constant, so translation was ignored)"
        )
    else:
        print("Mode: physics replay of recorded observation.state with fixed initial base pose")
    print(
        f"Actuators — body: {(body_ctrl_ids >= 0).sum()}/{len(body_jids)}, "
        f"left_hand: {(left_ctrl_ids >= 0).sum()}/{len(left_jids)}, "
        f"right_hand: {(right_ctrl_ids >= 0).sum()}/{len(right_jids)}"
    )

    viewer = None
    if not no_viewer:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
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

    dt = 1.0 / fps
    try:
        for i, target_state in enumerate(states):
            t_start = time.perf_counter()
            target_state = np.asarray(target_state, dtype=np.float64)
            if drive_recorded_base and base_pos_arr is not None and base_quat_arr is not None:
                i_next = min(i + 1, total_frames - 1)
                _physics_step(
                    model,
                    data,
                    body_ctrl_ids,
                    target_state[_BODY_IDX],
                    left_ctrl_ids,
                    target_state[_LEFT_HAND_IDX],
                    right_ctrl_ids,
                    target_state[_RIGHT_HAND_IDX],
                    n_steps=sim_substeps,
                    root_jid=root_jid if root_jid is not None else -1,
                    base_pos_start=base_pos_arr[i],
                    base_pos_end=base_pos_arr[i_next],
                    base_quat_start=base_quat_arr[i],
                    base_quat_end=base_quat_arr[i_next],
                )
            else:
                _physics_track_state(
                    model,
                    data,
                    body_jids,
                    left_jids,
                    right_jids,
                    body_ctrl_ids,
                    left_ctrl_ids,
                    right_ctrl_ids,
                    target_state,
                    sim_substeps,
                )
            mujoco.mj_forward(model, data)

            if viewer is not None and viewer.is_running():
                viewer.sync()

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, dt - elapsed))
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay LeRobot data in MuJoCo physics. By default this tracks "
            "recorded observation.state with PD control. With --sonic_encoder "
            "and --sonic_decoder, it uses SONIC policy replay in physics mode."
        )
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
        default=_SIM_STEPS_PER_POLICY,
        help="Physics substeps per dataset frame.",
    )
    parser.add_argument(
        "--drive_recorded_base",
        action="store_true",
        help="Drive robot.base_pos/robot.base_quat every physics step instead of leaving the base free.",
    )
    parser.add_argument("--sonic_encoder", default=None, help="Path to model_encoder.onnx.")
    parser.add_argument("--sonic_decoder", default=None, help="Path to model_decoder.onnx.")
    parser.add_argument("--compare", action="store_true", help="Enable SONIC comparison metrics.")
    parser.add_argument(
        "--upper_body_from_action",
        action="store_true",
        help="In SONIC mode, override waist and arm joints with recorded action targets.",
    )
    args = parser.parse_args()

    if args.sonic_encoder or args.sonic_decoder:
        if not (args.sonic_encoder and args.sonic_decoder):
            parser.error("--sonic_encoder and --sonic_decoder must be provided together")
        print("Loading SONIC models for physics replay...")
        sonic_runner = SonicRunner(
            args.sonic_encoder,
            args.sonic_decoder,
            fps=args.fps,
            closed_loop=False,
        )
        playback(
            dataset_dir=args.dataset_dir,
            episode=args.episode,
            env_name=args.env_name,
            output_video=None,
            cameras=["overview"],
            no_viewer=args.no_viewer,
            fps=args.fps,
            video_width=640,
            video_height=360,
            sonic_runner=sonic_runner,
            compare=args.compare,
            physics=True,
            upper_body_from_action=args.upper_body_from_action,
        )
        return

    playback_physics_state(
        dataset_dir=args.dataset_dir,
        episode=args.episode,
        env_name=args.env_name,
        fps=args.fps,
        no_viewer=args.no_viewer,
        sim_substeps=args.sim_substeps,
        drive_recorded_base=args.drive_recorded_base,
    )


if __name__ == "__main__":
    main()
