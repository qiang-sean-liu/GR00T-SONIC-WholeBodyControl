"""Export a recorded SONIC+PICO episode to an MP4 video file.

Each video frame contains (side by side):

    ┌─────────────────┬─────────────────┬──────────────────┐
    │  head_cam_left  │  head_cam_right │  SMPL skeleton   │
    │   (camera)      │    (camera)     │  (3-D view)      │
    └─────────────────┴─────────────────┴──────────────────┘

A HUD overlaid on the left panel shows frame index, elapsed time,
selected joint angles, and VR wrist positions.

Usage::

    # Write to episode_dir/playback.mp4 at recording fps
    python gear_sonic/scripts/playback_episode.py <episode_dir>

    # Custom output path and fps
    python gear_sonic/scripts/playback_episode.py <episode_dir> \\
        --output /tmp/demo.mp4 --fps 25

    # Skip skeleton (faster)
    python gear_sonic/scripts/playback_episode.py <episode_dir> --no_skeleton
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")           # off-screen rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


# ---------------------------------------------------------------------------
# SMPL skeleton (24 joints)
# ---------------------------------------------------------------------------

_SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
     9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]

_BONE_COLORS = {
    # spine / head
    (0, 3): "#cccccc", (3, 6): "#cccccc", (6, 9): "#cccccc",
    (9, 12): "#cccccc", (12, 15): "#cccccc",
    # left side
    (0, 1): "#4da6ff", (1, 4): "#4da6ff", (4, 7): "#4da6ff", (7, 10): "#4da6ff",
    (9, 13): "#4da6ff", (13, 16): "#4da6ff", (16, 18): "#4da6ff",
    (18, 20): "#4da6ff", (20, 22): "#4da6ff",
    # right side
    (0, 2): "#ff6b6b", (2, 5): "#ff6b6b", (5, 8): "#ff6b6b", (8, 11): "#ff6b6b",
    (9, 14): "#ff6b6b", (14, 17): "#ff6b6b", (17, 19): "#ff6b6b",
    (19, 21): "#ff6b6b", (21, 23): "#ff6b6b",
}

def _bone_color(p: int, c: int) -> str:
    return _BONE_COLORS.get((p, c), _BONE_COLORS.get((c, p), "#888888"))


# ---------------------------------------------------------------------------
# Skeleton renderer  (returns BGR uint8 array)
# ---------------------------------------------------------------------------

_fig: Optional[plt.Figure] = None
_ax:  Optional[plt.Axes]   = None


def _init_fig(h: int, w: int):
    global _fig, _ax
    dpi = 100
    _fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="#1a1a1a")
    _ax  = _fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")
    _fig.tight_layout(pad=0.4)


def render_skeleton(joints: np.ndarray, h: int, w: int) -> np.ndarray:
    """joints: (24, 3) in world coords (x-right, y-fwd, z-up).
    Returns (h, w, 3) BGR uint8."""
    global _fig, _ax
    if _fig is None:
        _init_fig(h, w)

    _ax.cla()
    _ax.set_facecolor("#1a1a1a")

    for child, parent in enumerate(_SMPL_PARENTS):
        if parent < 0:
            continue
        color = _bone_color(parent, child)
        _ax.plot([joints[parent, 0], joints[child, 0]],
                 [joints[parent, 1], joints[child, 1]],
                 [joints[parent, 2], joints[child, 2]],
                 color=color, linewidth=2.5)

    _ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                c="white", s=14, zorder=5)

    pelvis = joints[0]
    r = 0.85
    _ax.set_xlim(pelvis[0] - r, pelvis[0] + r)
    _ax.set_ylim(pelvis[1] - r, pelvis[1] + r)
    _ax.set_zlim(pelvis[2] - 0.1, pelvis[2] + 1.9)
    _ax.set_xlabel("x", color="#666", fontsize=7)
    _ax.set_ylabel("y", color="#666", fontsize=7)
    _ax.set_zlabel("z", color="#666", fontsize=7)
    _ax.tick_params(colors="#666", labelsize=6)
    for axis in [_ax.xaxis, _ax.yaxis, _ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#333333")
    _ax.view_init(elev=10, azim=-70)

    _fig.canvas.draw()
    buf  = np.frombuffer(_fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape(_fig.canvas.get_width_height()[::-1] + (4,))
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _put(img, text, y, scale=0.42):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),    3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, frame_idx: int, n_frames: int,
             t_rel: float, pico: dict, sonic: dict) -> np.ndarray:
    img = img.copy()
    fi  = min(frame_idx, n_frames - 1)
    _put(img, f"Frame {fi+1}/{n_frames}   t = {t_rel:.2f} s", 18)

    if "body_q_measured" in sonic and fi < len(sonic["body_q_measured"]):
        q = sonic["body_q_measured"][fi]
        _put(img, f"Meas   L_knee={q[3]:.2f}  R_knee={q[9]:.2f}"
                  f"  L_elbow={q[18]:.2f}  R_elbow={q[25]:.2f} rad", 35)

    if "body_q_target" in sonic and fi < len(sonic["body_q_target"]):
        q = sonic["body_q_target"][fi]
        _put(img, f"Target L_knee={q[3]:.2f}  R_knee={q[9]:.2f}"
                  f"  L_elbow={q[18]:.2f}  R_elbow={q[25]:.2f} rad", 52)

    if "vr_position" in pico and fi < len(pico["vr_position"]):
        p = pico["vr_position"][fi]
        _put(img, f"L-wrist ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
                  f"  R-wrist ({p[3]:.2f},{p[4]:.2f},{p[5]:.2f}) m", 69)
    return img


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode(episode_dir: str):
    ep = Path(episode_dir)
    with open(ep / "meta.json") as f:
        meta = json.load(f)
    pico  = dict(np.load(ep / "pico.npz"))
    sonic = dict(np.load(ep / "sonic.npz")) if (ep / "sonic.npz").exists() else {}

    cameras: Dict[str, List[Path]] = {}
    img_dir = ep / "images"
    if img_dir.exists():
        for cam_dir in sorted(img_dir.iterdir()):
            if cam_dir.is_dir():
                frames = sorted(cam_dir.glob("*.jpg"))
                if frames:
                    cameras[cam_dir.name] = frames

    return meta, pico, sonic, cameras


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Export a SONIC+PICO episode to MP4.")
    ap.add_argument("episode_dir")
    ap.add_argument("--output", default="",
                    help="Output video path (default: <episode_dir>/playback.mp4)")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="Video fps (0 = infer from recording timestamps)")
    ap.add_argument("--no_skeleton", action="store_true",
                    help="Omit the 3-D SMPL skeleton panel")
    ap.add_argument("--panel_height", type=int, default=480,
                    help="Height of each panel in pixels (default: 480)")
    args = ap.parse_args()

    meta, pico, sonic, cameras = load_episode(args.episode_dir)
    n_frames   = meta["n_frames"]
    duration   = meta["duration_s"]
    cam_names  = sorted(cameras.keys())
    smpl_joints = pico.get("smpl_joints")   # (T, 5, 24, 3) or None
    has_skeleton = smpl_joints is not None and not args.no_skeleton

    t_realtime = pico.get("timestamp_realtime", np.zeros((n_frames, 1))).flatten()
    t_start    = float(t_realtime[0]) if len(t_realtime) > 0 else 0.0

    fps = args.fps if args.fps > 0 else round(n_frames / max(duration, 1e-6), 1)
    fps = max(1.0, fps)

    out_path = args.output or str(Path(args.episode_dir) / "playback.mp4")

    # Determine panel dimensions from first camera image
    H = args.panel_height
    W = H * 4 // 3   # 4:3 default; overridden below if actual image differs
    if cam_names:
        probe = cv2.imread(str(cameras[cam_names[0]][0]))
        if probe is not None:
            H, W = probe.shape[:2]

    n_cam_panels = len(cam_names) if cam_names else 1
    n_panels     = n_cam_panels + (1 if has_skeleton else 0)
    total_w      = W * n_panels

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (total_w, H))
    if not writer.isOpened():
        print(f"[Playback] ERROR: could not open VideoWriter for {out_path}")
        sys.exit(1)

    print(f"Episode : {args.episode_dir}")
    print(f"Frames  : {n_frames}  ({duration:.1f} s  @{fps:.1f} fps)")
    print(f"Cameras : {cam_names}")
    print(f"Skeleton: {'yes' if has_skeleton else 'no'}")
    print(f"Output  : {out_path}")

    for fi in range(n_frames):
        panels = []

        # Camera panels
        for cam in (cam_names if cam_names else []):
            img_list = cameras[cam]
            idx      = min(fi, len(img_list) - 1)
            img      = cv2.imread(str(img_list[idx])) if idx >= 0 else None
            if img is None:
                img = np.zeros((H, W, 3), dtype=np.uint8)
            elif img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            panels.append(img)

        if not panels:
            panels.append(np.zeros((H, W, 3), dtype=np.uint8))

        # HUD on leftmost panel
        t_rel    = float(t_realtime[min(fi, len(t_realtime) - 1)]) - t_start
        panels[0] = draw_hud(panels[0], fi, n_frames, t_rel, pico, sonic)

        # Skeleton panel
        if has_skeleton:
            joints   = smpl_joints[min(fi, len(smpl_joints) - 1), -1]  # (24, 3)
            skel_img = render_skeleton(joints, H, W)
            if skel_img.shape[:2] != (H, W):
                skel_img = cv2.resize(skel_img, (W, H))
            panels.append(skel_img)

        frame = np.hstack(panels)
        writer.write(frame)

        if fi % 50 == 0 or fi == n_frames - 1:
            pct = 100 * (fi + 1) / n_frames
            print(f"  {fi+1}/{n_frames} ({pct:.0f}%)", end="\r", flush=True)

    writer.release()
    if _fig is not None:
        plt.close(_fig)

    # Re-encode with ffmpeg for broad player compatibility (H.264 + yuv420p)
    tmp_path = out_path + ".tmp.mp4"
    os.rename(out_path, tmp_path)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
         out_path],
        capture_output=True, text=True,
    )
    os.unlink(tmp_path)
    if result.returncode != 0:
        print(f"[Playback] ffmpeg re-encode failed:\n{result.stderr}")
        sys.exit(1)

    print(f"\n[Playback] Saved: {out_path}")


if __name__ == "__main__":
    main()
