"""Replay a LeRobot episode in MuJoCo by directly feeding recorded 43-DOF joint positions.

The script kinematically replays observation.state from a LeRobot Parquet file — it
bypasses PD control and sets qpos directly, then calls mj_forward() to update body
poses.  No unitree_sdk2py or WBC process is needed.

43-DOF layout (matches convert_sonic_to_lerobot.py):
    state[0:6]   left_leg      (6)
    state[6:12]  right_leg     (6)
    state[12:15] waist         (3)
    state[15:22] left_arm      (7)
    state[22:29] left_hand     (7)
    state[29:36] right_arm     (7)
    state[36:43] right_hand    (7)

Usage:
    # Onscreen viewer + save third-person MP4:
    conda run -n sonic_dc python gear_sonic/scripts/playback_lerobot.py \\
        --dataset_dir /home/horizon/wrk/SONIC/lerobot_dataset \\
        --episode 0 \\
        --env_name kitchen_pnp_apple \\
        --output_video playback_ep0.mp4

    # Headless (no viewer) — just save video:
    conda run -n sonic_dc python gear_sonic/scripts/playback_lerobot.py \\
        --dataset_dir /home/horizon/wrk/SONIC/lerobot_dataset \\
        --episode 0 \\
        --env_name kitchen_pnp_apple \\
        --output_video playback_ep0.mp4 \\
        --no_viewer

    # Different camera (default: "overview"):
    conda run -n sonic_dc python gear_sonic/scripts/playback_lerobot.py ... \\
        --camera overview

Environment → XML mapping:
    kitchen_pnp_apple  decoupled_wbc/control/robot_model/model_data/g1/kitchen_pnp_apple_43dof.xml
    pnp_cube           decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml
    lift_box           decoupled_wbc/control/robot_model/model_data/g1/lift_box_43dof.xml
    pnp_bottle         decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml
    default            decoupled_wbc/control/robot_model/model_data/g1/scene_43dof.xml
"""

import argparse
import os
import pathlib
import time

import av
import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_ENV_XML = {
    "kitchen_pnp_apple": "decoupled_wbc/control/robot_model/model_data/g1/kitchen_pnp_apple_43dof.xml",
    "pnp_cube":          "decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml",
    "lift_box":          "decoupled_wbc/control/robot_model/model_data/g1/lift_box_43dof.xml",
    "pnp_bottle":        "decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml",
    "default":           "decoupled_wbc/control/robot_model/model_data/g1/scene_43dof.xml",
}

# 43-DOF slice → body (29) de-assembly
_BODY_IDX = np.concatenate([np.arange(22), np.arange(29, 36)])  # 29 joints
_LEFT_HAND_IDX = np.arange(22, 29)   # 7 joints
_RIGHT_HAND_IDX = np.arange(36, 43)  # 7 joints

# Joint name substrings that identify body / hand joints (matches base_sim.py)
_BODY_JOINT_KEYS = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]

# Default root pose: pelvis height for a standing robot (metres)
_ROOT_HEIGHT = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_xml(env_name: str) -> mujoco.MjModel:
    rel = _ENV_XML.get(env_name)
    if rel is None:
        raise ValueError(f"Unknown env_name '{env_name}'. Choose from: {list(_ENV_XML)}")
    xml_path = str(_REPO_ROOT / rel)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    return mujoco.MjModel.from_xml_path(xml_path)


def _build_joint_indices(model: mujoco.MjModel):
    """Return (body_joint_index, left_hand_index, right_hand_index) — arrays of MuJoCo joint ids."""
    body, left_hand, right_hand = [], [], []
    for i in range(model.njnt):
        name = model.joint(i).name
        if any(k in name for k in _BODY_JOINT_KEYS):
            body.append(i)
        elif "left_hand" in name:
            left_hand.append(i)
        elif "right_hand" in name:
            right_hand.append(i)
    return np.array(body), np.array(left_hand), np.array(right_hand)


def _load_episode(dataset_dir: str, episode: int):
    """Load episode data from Parquet.

    Returns:
        states: float32 [T, 43] joint positions
        task_indices: list of task_index per frame
        base_pos: float64 [T, 3] or None if not in dataset
        base_quat: float64 [T, 4] wxyz or None if not in dataset
    """
    chunk = episode // 1000
    path = os.path.join(
        dataset_dir,
        f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")

    cols = ["observation.state", "task_index"]
    schema = pq.read_schema(path)
    has_base = "robot.base_pos" in schema.names and "robot.base_quat" in schema.names
    if has_base:
        cols += ["robot.base_pos", "robot.base_quat"]

    table = pq.read_table(path, columns=cols)
    state_col = table.column("observation.state")
    states = np.array([row.as_py() for row in state_col], dtype=np.float32)
    task_indices = table.column("task_index").to_pylist()

    base_pos = base_quat = None
    if has_base:
        base_pos = np.array([row.as_py() for row in table.column("robot.base_pos")], dtype=np.float64)
        base_quat = np.array([row.as_py() for row in table.column("robot.base_quat")], dtype=np.float64)

    return states, task_indices, base_pos, base_quat


def _set_qpos(data: mujoco.MjData, model: mujoco.MjModel,
               body_jids, left_jids, right_jids, state43: np.ndarray,
               root_jid: int | None = None,
               base_pos: np.ndarray | None = None,
               base_quat: np.ndarray | None = None):
    """Set MuJoCo qpos from a 43-DOF joint vector (kinematic, no physics).

    Optionally sets the free-joint root pose from base_pos (xyz) and
    base_quat (wxyz, MuJoCo convention).
    """
    body29 = state43[_BODY_IDX]
    left7 = state43[_LEFT_HAND_IDX]
    right7 = state43[_RIGHT_HAND_IDX]
    # jnt_qposadr[i] = start index of joint i in qpos
    data.qpos[model.jnt_qposadr[body_jids]] = body29
    if len(left_jids):
        data.qpos[model.jnt_qposadr[left_jids]] = left7
    if len(right_jids):
        data.qpos[model.jnt_qposadr[right_jids]] = right7
    # Root pose (free joint): qpos[adr:adr+3] = xyz, qpos[adr+3:adr+7] = wxyz
    if root_jid is not None and base_pos is not None and base_quat is not None:
        adr = model.jnt_qposadr[root_jid]
        data.qpos[adr:adr + 3] = base_pos
        data.qpos[adr + 3:adr + 7] = base_quat


def _make_video_writer(path: str, width: int, height: int, fps: float):
    container = av.open(path, "w")
    stream = container.add_stream("h264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    return container, stream


# ---------------------------------------------------------------------------
# Main playback loop
# ---------------------------------------------------------------------------

def playback(
    dataset_dir: str,
    episode: int,
    env_name: str,
    output_video: str | None,
    camera: str,
    no_viewer: bool,
    fps: float,
    video_width: int,
    video_height: int,
):
    # 1. Load data
    states, task_indices, base_pos, base_quat = _load_episode(dataset_dir, episode)
    T = len(states)
    task_index = task_indices[0] if task_indices else None
    if base_pos is not None:
        print(f"Root pose loaded: base_pos x range [{base_pos[:,0].min():.3f}, {base_pos[:,0].max():.3f}] m")
    else:
        print("Warning: robot.base_pos/base_quat not in dataset — root fixed at (0,0,0.8).")

    # Resolve task description from tasks.jsonl
    task_desc = "(unknown)"
    tasks_file = os.path.join(dataset_dir, "meta/tasks.jsonl")
    if os.path.exists(tasks_file):
        import json
        with open(tasks_file) as f:
            for line in f:
                rec = json.loads(line)
                if task_index is not None and rec.get("task_index") == task_index:
                    task_desc = rec.get("task", task_desc)
                    break

    print(f"Episode {episode}: {T} frames @ {fps:.0f} Hz")
    print(f"Task [{task_index}]: {task_desc}")

    # 2. Load MuJoCo model
    model = _load_xml(env_name)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Locate the free (floating-base) joint
    root_jid_list = [i for i in range(model.njnt) if model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE]
    root_jid = root_jid_list[0] if root_jid_list else None
    if root_jid is not None and base_pos is None:
        # No recorded root pose — fix the robot at a standing height
        adr = model.jnt_qposadr[root_jid]
        data.qpos[adr + 2] = _ROOT_HEIGHT  # z = height
        data.qpos[adr + 3] = 1.0           # quaternion w=1 (identity)

    body_jids, left_jids, right_jids = _build_joint_indices(model)
    print(f"Joints found — body: {len(body_jids)}, left_hand: {len(left_jids)}, right_hand: {len(right_jids)}")

    # 3. Verify the camera exists (fallback to free camera)
    cam_names = [model.cam(i).name for i in range(model.ncam)]
    print(f"Scene cameras: {cam_names}")
    if camera not in cam_names:
        print(f"  Warning: camera '{camera}' not found. Falling back to 'overview' or first camera.")
        camera = cam_names[0] if cam_names else None

    # 4. Set up offscreen renderer for video
    renderer = None
    container = stream = None
    if output_video:
        # Resize the model's offscreen framebuffer to match requested video dimensions
        model.vis.global_.offwidth = video_width
        model.vis.global_.offheight = video_height
        renderer = mujoco.Renderer(model, height=video_height, width=video_width)
        container, stream = _make_video_writer(output_video, video_width, video_height, fps)
        print(f"Saving video → {output_video}  ({video_width}×{video_height} @ {fps:.0f} fps)")

    # 5. Launch onscreen viewer (passive, non-blocking)
    viewer = None
    if not no_viewer:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        if viewer is not None:
            # Third-person tracking: track pelvis
            try:
                pelvis_id = model.body("pelvis").id
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = pelvis_id
                viewer.cam.distance = 2.5
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 135
            except Exception:
                pass

    # 6. Replay loop
    dt = 1.0 / fps
    try:
        for i, state in enumerate(states):
            t_start = time.perf_counter()

            bp = base_pos[i] if base_pos is not None else None
            bq = base_quat[i] if base_quat is not None else None
            _set_qpos(data, model, body_jids, left_jids, right_jids, state,
                      root_jid=root_jid, base_pos=bp, base_quat=bq)
            mujoco.mj_forward(model, data)  # compute kinematics (no dynamics)

            # Render offscreen → encode video frame
            if renderer is not None and container is not None:
                if camera is not None:
                    renderer.update_scene(data, camera=camera)
                else:
                    renderer.update_scene(data)
                rgb = renderer.render()  # uint8 HWC
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = i
                for pkt in stream.encode(frame):
                    container.mux(pkt)

            # Sync onscreen viewer
            if viewer is not None and viewer.is_running():
                viewer.sync()

            # Pace to real-time
            elapsed = time.perf_counter() - t_start
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

            if (i + 1) % 100 == 0 or i == T - 1:
                print(f"  Frame {i+1}/{T}")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if viewer is not None:
            viewer.close()
        if container is not None:
            # Flush encoder
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            print(f"Video saved: {output_video}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode in MuJoCo.")
    parser.add_argument("--dataset_dir", required=True, help="Path to LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (default: 0).")
    parser.add_argument(
        "--env_name",
        default="kitchen_pnp_apple",
        choices=list(_ENV_XML),
        help="MuJoCo scene to load (default: kitchen_pnp_apple).",
    )
    parser.add_argument(
        "--output_video",
        default=None,
        help="Path to save third-person MP4 (e.g. playback.mp4). Omit to skip.",
    )
    parser.add_argument(
        "--camera",
        default="overview",
        help="Camera name for video render (default: overview).",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Disable onscreen MuJoCo viewer (useful for headless rendering).",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Playback FPS (default: 20).")
    parser.add_argument("--video_width", type=int, default=1280, help="Video width (default: 1280).")
    parser.add_argument("--video_height", type=int, default=720, help="Video height (default: 720).")
    args = parser.parse_args()

    playback(
        dataset_dir=args.dataset_dir,
        episode=args.episode,
        env_name=args.env_name,
        output_video=args.output_video,
        camera=args.camera,
        no_viewer=args.no_viewer,
        fps=args.fps,
        video_width=args.video_width,
        video_height=args.video_height,
    )


if __name__ == "__main__":
    main()
