"""Convert SONIC recordings to LeRobot using q_target_cmd as action body source.

This is a wrapper around convert_sonic_to_lerobot.py.
It keeps the original converter untouched and only changes one input field:

  sonic.body_q_target <- sonic.q_target_cmd

for each episode in a temporary input directory, then runs the original
converter on that temporary directory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _import_base_converter():
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir / "convert_sonic_to_lerobot.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("convert_sonic_to_lerobot", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base converter: {base_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, base_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SONIC NPZ recordings using q_target_cmd as action body source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--task", default=None)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--no_images", action="store_true")
    p.add_argument("--append", action="store_true")
    p.add_argument("--robot_type", default="g1")
    return p.parse_args()


def _prepare_episode(ep_dir: Path, out_ep_dir: Path) -> None:
    out_ep_dir.mkdir(parents=True, exist_ok=True)

    # Copy lightweight metadata files when present.
    for name in ("meta.json", "image_timestamps.npz"):
        src = ep_dir / name
        if src.exists():
            shutil.copy2(src, out_ep_dir / name)

    # Symlink images directory to avoid large copies.
    images_src = ep_dir / "images"
    if images_src.exists():
        (out_ep_dir / "images").symlink_to(images_src.resolve(), target_is_directory=True)

    # Keep pico.npz unchanged.
    shutil.copy2(ep_dir / "pico.npz", out_ep_dir / "pico.npz")

    # Rewrite sonic.npz: body_q_target <- q_target_cmd
    sonic_in = np.load(ep_dir / "sonic.npz")
    if "q_target_cmd" not in sonic_in:
        raise ValueError(f"{ep_dir}: missing q_target_cmd in sonic.npz")

    out_data = {k: sonic_in[k] for k in sonic_in.files}
    qcmd = out_data["q_target_cmd"]
    out_data["body_q_target"] = qcmd.astype(np.float64, copy=True)
    np.savez_compressed(out_ep_dir / "sonic.npz", **out_data)


def main() -> None:
    args = _parse_args()
    base_mod, base_script = _import_base_converter()

    episodes = base_mod._find_episodes(args.input_dir)
    print(f"Found {len(episodes)} episode(s) in {args.input_dir}")

    with tempfile.TemporaryDirectory(prefix="qcmd_convert_") as td:
        tmp_root = Path(td)
        for ep in episodes:
            tmp_ep = tmp_root / ep.name
            _prepare_episode(ep, tmp_ep)

        cmd = [
            sys.executable,
            str(base_script),
            "--input_dir",
            str(tmp_root),
            "--output_dir",
            str(args.output_dir),
            "--fps",
            str(args.fps),
            "--robot_type",
            str(args.robot_type),
        ]
        if args.task:
            cmd.extend(["--task", args.task])
        if args.no_images:
            cmd.append("--no_images")
        if args.append:
            cmd.append("--append")

        print("Running base converter with q_target_cmd-mapped body_q_target...")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
