"""Compare recorded decoder tensors with the C++ PolicyEngine TensorRT path.

This wrapper extracts `sonic.decoder_obs`, `sonic.decoder_action_raw`, and
`sonic.q_target_cmd` from a LeRobot parquet file into a small binary file, then
invokes the standalone C++ `replay_policy_engine_decoder_test` executable.

The C++ executable reuses the deploy `PolicyEngine` path without touching the
teleop binary.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


MAGIC = b"SONICDEC"
VERSION = 1
DECODER_OBS_DIM = 994
ACTION_DIM = 29


def episode_path(dataset_dir: Path, episode: int) -> Path:
    chunk = episode // 1000
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"


def load_array(table: pq.Table, name: str, width: int, dtype: np.dtype) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"Missing required column {name!r}")
    arr = np.asarray(table[name].to_pylist(), dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{name} must have shape (T, {width}); got {arr.shape}")
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{name} contains non-finite values, first bad index {bad[0].tolist()}")
    return np.ascontiguousarray(arr)


def write_binary(path: Path, decoder_obs: np.ndarray, action_raw: np.ndarray, q_target: np.ndarray) -> None:
    frames = decoder_obs.shape[0]
    header = struct.pack(
        "<8sIIQQQ",
        MAGIC,
        VERSION,
        0,
        frames,
        DECODER_OBS_DIM,
        ACTION_DIM,
    )
    with path.open("wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(decoder_obs, dtype=np.float32).tobytes(order="C"))
        f.write(np.ascontiguousarray(action_raw, dtype=np.float32).tobytes(order="C"))
        f.write(np.ascontiguousarray(q_target, dtype=np.float64).tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("target/release/replay_policy_engine_decoder_test"),
        help="Built C++ PolicyEngine replay test executable.",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="Use PolicyEngine FP16, matching --policy-precision 16.")
    parser.add_argument("--keep_input_bin", type=Path, default=None)
    args = parser.parse_args()

    parquet_path = episode_path(args.dataset_dir, args.episode)
    table = pq.read_table(
        parquet_path,
        columns=["sonic.decoder_obs", "sonic.decoder_action_raw", "sonic.q_target_cmd"],
    )

    decoder_obs = load_array(table, "sonic.decoder_obs", DECODER_OBS_DIM, np.float32)
    action_raw = load_array(table, "sonic.decoder_action_raw", ACTION_DIM, np.float32)
    q_target = load_array(table, "sonic.q_target_cmd", ACTION_DIM, np.float64)

    start = int(np.clip(args.start_frame, 0, len(decoder_obs)))
    end = len(decoder_obs) if args.num_frames is None else min(len(decoder_obs), start + max(0, args.num_frames))
    if start >= end:
        raise ValueError(f"Empty frame range: start={start}, end={end}, total={len(decoder_obs)}")

    decoder_obs = decoder_obs[start:end]
    action_raw = action_raw[start:end]
    q_target = q_target[start:end]

    if not args.executable.is_file():
        raise FileNotFoundError(
            f"C++ test executable not found: {args.executable}\n"
            "Build it first, for example:\n"
            "  cmake --build gear_sonic_deploy/build --target replay_policy_engine_decoder_test"
        )

    with tempfile.TemporaryDirectory(prefix="sonic_decoder_policy_engine_") as tmpdir:
        input_bin = args.keep_input_bin or Path(tmpdir) / "decoder_test.bin"
        write_binary(input_bin, decoder_obs, action_raw, q_target)

        cmd = [
            str(args.executable),
            "--model",
            str(args.model),
            "--input",
            str(input_bin),
        ]
        if args.fp16:
            cmd.append("--fp16")

        print(f"Parquet: {parquet_path}")
        print(f"Frames: {start}..{end - 1} ({end - start})")
        print(f"Executable: {args.executable}")
        print(f"Model: {args.model}")
        print(f"Input bin: {input_bin}")
        print()
        subprocess.run(cmd, check=True)

        if args.keep_input_bin:
            print(f"\nKept input binary: {input_bin}")
        elif os.path.exists(input_bin):
            # Should only happen if input_bin was outside tempdir.
            os.remove(input_bin)


if __name__ == "__main__":
    main()
