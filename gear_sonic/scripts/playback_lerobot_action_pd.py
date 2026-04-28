"""Replay recorded SONIC/WBC actions through MuJoCo PD control.

This is action-based playback: it loads ``action.wbc`` from a LeRobot episode
and uses those values as joint-position targets for the same joint-space PD
controller used by the SONIC playback path. Unlike pure state playback, it does
not set every frame's joint qpos from ``observation.state``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq

from gear_sonic.scripts.playback_lerobot import (
    _BODY_IDX,
    _LEFT_HAND_IDX,
    _RIGHT_HAND_IDX,
    _ROOT_HEIGHT,
    _SIM_STEPS_PER_POLICY,
    _build_ctrl_map,
    _build_joint_indices,
    _load_xml,
    _physics_step,
    _set_qpos,
)


def _episode_path(dataset_dir: str, episode: int) -> str:
    chunk = episode // 1000
    return os.path.join(
        dataset_dir,
        f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
    )


def _load_action_episode(dataset_dir: str, episode: int):
    path = _episode_path(dataset_dir, episode)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")

    schema = pq.read_schema(path)
    action_col = "action.wbc" if "action.wbc" in schema.names else "action"
    if action_col not in schema.names:
        raise ValueError("Expected an action column named 'action.wbc' or 'action'.")

    cols = ["observation.state", action_col, "task_index"]
    has_base_pos = "robot.base_pos" in schema.names
    base_quat_col = (
        "robot.base_quat"
        if "robot.base_quat" in schema.names
        else "observation.root_orientation"
        if "observation.root_orientation" in schema.names
        else None
    )
    if has_base_pos:
        cols.append("robot.base_pos")
    if base_quat_col is not None:
        cols.append(base_quat_col)

    table = pq.read_table(path, columns=cols)
    states = np.array([r.as_py() for r in table.column("observation.state")], dtype=np.float64)
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
    return states, actions, task_indices, base_pos, base_quat


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
    sim_substeps: int,
    free_base: bool,
    debug: bool,
) -> None:
    states, actions, task_indices, base_pos, base_quat = _load_action_episode(dataset_dir, episode)
    total_frames = len(actions)
    task_index = task_indices[0] if task_indices else None

    model = _load_xml(env_name)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    root_jids = [i for i in range(model.njnt) if model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE]
    root_jid = root_jids[0] if root_jids else None

    body_jids, left_jids, right_jids = _build_joint_indices(model)
    body_ctrl_ids = _build_ctrl_map(model, body_jids)
    left_ctrl_ids = _build_ctrl_map(model, left_jids)
    right_ctrl_ids = _build_ctrl_map(model, right_jids)

    init_base_pos = (
        base_pos[0]
        if base_pos is not None
        else np.array([0.0, 0.0, _ROOT_HEIGHT], dtype=np.float64)
    )
    init_base_quat = (
        base_quat[0]
        if base_quat is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
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

    print(f"Episode {episode}: {total_frames} action frames @ {fps:.0f} Hz")
    print(f"Task [{task_index}]: {_task_description(dataset_dir, task_index)}")
    print(f"Environment: {env_name}")
    print(
        "Mode: action.wbc playback through joint-space PD "
        f"({'free floating base' if free_base else 'recorded base driven when available'})"
    )
    print(
        f"Actuators - body: {(body_ctrl_ids >= 0).sum()}/{len(body_jids)}, "
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
        for i, action in enumerate(actions):
            t_start = time.perf_counter()
            body29_target = action[_BODY_IDX]
            left7_target = action[_LEFT_HAND_IDX]
            right7_target = action[_RIGHT_HAND_IDX]

            i_next = min(i + 1, total_frames - 1)
            use_base_drive = (not free_base) and base_pos is not None and base_quat is not None
            _physics_step(
                model,
                data,
                body_ctrl_ids,
                body29_target,
                left_ctrl_ids,
                left7_target,
                right_ctrl_ids,
                right7_target,
                n_steps=sim_substeps,
                root_jid=root_jid if use_base_drive and root_jid is not None else -1,
                base_pos_start=base_pos[i] if use_base_drive else None,
                base_pos_end=base_pos[i_next] if use_base_drive else None,
                base_quat_start=base_quat[i] if use_base_drive else None,
                base_quat_end=base_quat[i_next] if use_base_drive else None,
            )
            mujoco.mj_forward(model, data)

            if viewer is not None and viewer.is_running():
                viewer.sync()

            if debug and (i < 10 or (i + 1) % 100 == 0 or i == total_frames - 1):
                sim_body29 = data.qpos[model.jnt_qposadr[body_jids]]
                root_msg = ""
                if use_base_drive and root_jid is not None:
                    root_adr = model.jnt_qposadr[root_jid]
                    root_msg = (
                        f" base={np.round(data.qpos[root_adr:root_adr + 3], 4)}"
                        f" rec_base={np.round(base_pos[i_next], 4)}"
                        f" quat={np.round(data.qpos[root_adr + 3:root_adr + 7], 4)}"
                    )
                print(
                    f"  Frame {i + 1}/{total_frames} "
                    f"target_leg={np.round(action[:12], 4)} "
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded action.wbc through the SONIC joint-space PD controller."
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
        help="MuJoCo substeps per recorded action frame.",
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
    )


if __name__ == "__main__":
    main()
