"""Replay LeRobot episode with action trajectory as simulator state.

Thin wrapper over playback_lerobot.py:
- keeps original playback implementation and camera/viewer behavior
- only swaps `states <- actions` when loading episode in non-SONIC mode
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _import_playback_module(repo_root: Path):
    mod_path = repo_root / "gear_sonic/scripts/playback_lerobot.py"
    spec = importlib.util.spec_from_file_location("playback_lerobot", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pb = _import_playback_module(repo_root)

    parser = argparse.ArgumentParser(description="Replay a LeRobot episode using recorded action trajectory.")
    parser.add_argument("--dataset_dir", required=True, help="Path to LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--env_name",
        default="kitchen_pnp_apple",
        choices=list(pb._ENV_XML),
        help="MuJoCo scene (default: kitchen_pnp_apple).",
    )
    parser.add_argument("--output_video", default=None)
    parser.add_argument("--camera", default=None, help="Single camera name (legacy shorthand).")
    parser.add_argument("--cameras", nargs="+", default=None, help="One or more camera names.")
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=360)
    args = parser.parse_args()

    if args.cameras:
        cameras = args.cameras
    elif args.camera:
        cameras = [args.camera]
    else:
        cameras = ["overview"]

    original_load_episode = pb._load_episode

    def _load_episode_action_as_state(dataset_dir: str, episode: int, sonic: bool = False):
        (
            states,
            actions,
            task_indices,
            base_pos,
            base_quat,
            base_ang_vel,
            body_dq,
            enc_obs_rec,
            token_rec,
            dec_obs_rec,
            dec_action_rec,
            q_target_cmd,
            smpl_joints,
            body_root_quat,
        ) = original_load_episode(dataset_dir, episode, sonic=sonic)
        if sonic:
            raise ValueError("playback_lerobot_action.py is non-SONIC only; do not pass policy flags.")
        states = actions.astype(states.dtype, copy=True)
        return (
            states,
            actions,
            task_indices,
            base_pos,
            base_quat,
            base_ang_vel,
            body_dq,
            enc_obs_rec,
            token_rec,
            dec_obs_rec,
            dec_action_rec,
            q_target_cmd,
            smpl_joints,
            body_root_quat,
        )

    setattr(pb, "_load_episode", _load_episode_action_as_state)

    pb.playback(
        dataset_dir=args.dataset_dir,
        episode=args.episode,
        env_name=args.env_name,
        output_video=args.output_video,
        cameras=cameras,
        no_viewer=args.no_viewer,
        fps=args.fps,
        video_width=args.video_width,
        video_height=args.video_height,
        sonic_runner=None,
        compare=False,
        physics=False,
        upper_body_from_action=False,
    )


if __name__ == "__main__":
    main()
