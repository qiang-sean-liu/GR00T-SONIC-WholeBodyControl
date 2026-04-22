"""Extract SMPL-mode encoder blocks from recorded SONIC model I/O.

Reads sonic.npz from a teleop episode directory and extracts:
  - smpl_joints_10frame_step1           (encoder_obs[:,  922:1642])
  - smpl_anchor_orientation_10frame_step1 (encoder_obs[:, 1642:1702])
  - motion_joint_positions_wrists_10frame_step1 (encoder_obs[:, 1702:1762])

Optionally runs model_encoder.onnx on recorded encoder_obs and compares against
recorded token_state to verify exact replay fidelity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


_ENC_LAYOUT = {
    "smpl_joints_10frame_step1": (922, 1642),
    "smpl_anchor_orientation_10frame_step1": (1642, 1702),
    "motion_joint_positions_wrists_10frame_step1": (1702, 1762),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract SMPL-mode encoder blocks from recordings/*/sonic.npz",
    )
    p.add_argument(
        "--episode_dir",
        required=True,
        help="Episode directory containing sonic.npz (e.g. recordings/20260410_122727_ep0002).",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output NPZ path (default: <episode_dir>/encoder_smpl_mode_blocks.npz). "
            "The file stores full [T, *] arrays for each block."
        ),
    )
    p.add_argument(
        "--encoder_model",
        default=None,
        help=(
            "Optional path to model_encoder.onnx for token verification. "
            "If provided and token_state exists, runs encoder_obs through ONNX."
        ),
    )
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index for a short console preview (default: 0).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    sonic_path = episode_dir / "sonic.npz"
    if not sonic_path.exists():
        raise FileNotFoundError(f"Missing sonic.npz: {sonic_path}")

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else episode_dir / "encoder_smpl_mode_blocks.npz"
    )

    data = np.load(sonic_path, allow_pickle=True)
    if "encoder_obs" not in data.files:
        raise KeyError(f"{sonic_path} does not contain key 'encoder_obs'")

    encoder_obs = data["encoder_obs"].astype(np.float64)
    if encoder_obs.ndim != 2 or encoder_obs.shape[1] < 1762:
        raise ValueError(f"Unexpected encoder_obs shape: {encoder_obs.shape}, expected [T,1762]")

    blocks = {}
    for name, (s, e) in _ENC_LAYOUT.items():
        blocks[name] = encoder_obs[:, s:e]

    np.savez_compressed(
        out_path,
        **blocks,
        encoder_obs=encoder_obs,
    )

    frame = int(np.clip(args.frame, 0, encoder_obs.shape[0] - 1))
    print(f"Loaded: {sonic_path}")
    print(f"Saved:  {out_path}")
    print(f"T={encoder_obs.shape[0]} frames, frame preview index={frame}")
    for name in _ENC_LAYOUT:
        arr = blocks[name]
        print(f"  {name:45s} shape={arr.shape}, frame{frame}_norm={np.linalg.norm(arr[frame]):.6f}")

    meta = {
        "source_sonic_npz": str(sonic_path),
        "output_npz": str(out_path),
        "num_frames": int(encoder_obs.shape[0]),
        "layout": {k: [int(v[0]), int(v[1])] for k, v in _ENC_LAYOUT.items()},
    }

    if args.encoder_model:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed in the active environment. "
                "Install it or rerun without --encoder_model."
            )

        enc_model = Path(args.encoder_model).expanduser().resolve()
        if not enc_model.exists():
            raise FileNotFoundError(f"Missing encoder model: {enc_model}")

        if "token_state" not in data.files:
            raise KeyError(f"{sonic_path} does not contain key 'token_state'")

        token_rec = data["token_state"].astype(np.float32)
        sess = ort.InferenceSession(str(enc_model), providers=["CPUExecutionProvider"])
        token_pred = np.stack(
            [
                sess.run(None, {"obs_dict": encoder_obs[i : i + 1].astype(np.float32)})[0][0]
                for i in range(encoder_obs.shape[0])
            ]
        )
        l2 = np.linalg.norm(token_pred - token_rec, axis=1)
        meta["encoder_model"] = str(enc_model)
        meta["token_l2_mean"] = float(l2.mean())
        meta["token_l2_max"] = float(l2.max())
        meta["token_l2_frame0"] = float(l2[0])
        print(
            "Token check: "
            f"mean={meta['token_l2_mean']:.9e}, "
            f"max={meta['token_l2_max']:.9e}, "
            f"frame0={meta['token_l2_frame0']:.9e}"
        )

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
