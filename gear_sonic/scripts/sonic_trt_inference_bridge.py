"""Python client for the standalone C++ TensorRT SONIC inference bridge."""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


_READY = struct.Struct("<6I")
_REQUEST = struct.Struct("<cI")
_RESPONSE = struct.Struct("<cI")
_READY_MAGIC = 0x31545254
_PROTOCOL_VERSION = 1


def _read_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise RuntimeError("TensorRT bridge closed before sending a complete response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_exact(stream: Any, data: bytes) -> None:
    stream.write(data)
    stream.flush()


class _TrtSessionAdapter:
    def __init__(self, bridge: "SonicTrtInferenceBridge", request_type: bytes, input_dim: int, output_dim: int):
        self._bridge = bridge
        self._request_type = request_type
        self.input_dim = input_dim
        self.output_dim = output_dim

    def run(self, _outputs: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if "obs_dict" not in inputs:
            raise KeyError("TensorRT bridge expects input key 'obs_dict'")
        obs = np.asarray(inputs["obs_dict"], dtype=np.float32)
        had_batch = obs.ndim == 2
        if had_batch:
            if obs.shape[0] != 1:
                raise ValueError(f"TensorRT bridge only supports batch size 1, got {obs.shape}")
            obs_1d = np.ascontiguousarray(obs[0], dtype=np.float32)
        else:
            obs_1d = np.ascontiguousarray(obs.reshape(-1), dtype=np.float32)

        if obs_1d.size != self.input_dim:
            raise ValueError(f"TensorRT bridge expected {self.input_dim} inputs, got {obs_1d.size}")

        out = self._bridge.infer(self._request_type, obs_1d, self.output_dim)
        return [out[np.newaxis, :] if had_batch else out]


class SonicTrtInferenceBridge:
    """Persistent subprocess bridge to C++ EncoderEngine + PolicyEngine TensorRT."""

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        *,
        executable: str | Path = "gear_sonic_deploy/target/release/sonic_trt_inference_bridge",
        encoder_fp16: bool = False,
        decoder_fp16: bool = False,
    ):
        read_fd, write_fd = os.pipe()
        cmd = [
            str(executable),
            "--encoder",
            str(encoder_path),
            "--decoder",
            str(decoder_path),
            "--response_fd",
            str(write_fd),
        ]
        if encoder_fp16:
            cmd.append("--encoder_fp16")
        if decoder_fp16:
            cmd.append("--decoder_fp16")

        self._read_fd = read_fd
        self._write_fd = write_fd
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        self._write_fd = -1

        ready = _READY.unpack(_read_exact(read_fd, _READY.size))
        magic, version, enc_in_dim, token_dim, dec_in_dim, action_dim = ready
        if magic != _READY_MAGIC or version != _PROTOCOL_VERSION:
            self.close()
            raise RuntimeError(f"Invalid TensorRT bridge ready header: magic={magic:#x} version={version}")

        self.encoder_input_dim = int(enc_in_dim)
        self.token_dim = int(token_dim)
        self.decoder_input_dim = int(dec_in_dim)
        self.action_dim = int(action_dim)
        self.encoder_session = _TrtSessionAdapter(self, b"E", self.encoder_input_dim, self.token_dim)
        self.decoder_session = _TrtSessionAdapter(self, b"D", self.decoder_input_dim, self.action_dim)

    def infer(self, request_type: bytes, obs: np.ndarray, output_dim: int) -> np.ndarray:
        if self._proc.stdin is None:
            raise RuntimeError("TensorRT bridge stdin is closed")
        payload = _REQUEST.pack(request_type, int(obs.size)) + obs.tobytes(order="C")
        _write_exact(self._proc.stdin, payload)
        response_type, count = _RESPONSE.unpack(_read_exact(self._read_fd, _RESPONSE.size))
        if response_type != request_type:
            raise RuntimeError(f"TensorRT bridge response type mismatch: {response_type!r} != {request_type!r}")
        if count != output_dim:
            raise RuntimeError(f"TensorRT bridge output dim mismatch: {count} != {output_dim}")
        raw = _read_exact(self._read_fd, int(count) * np.dtype(np.float32).itemsize)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.stdin is not None and proc.poll() is None:
            try:
                _write_exact(proc.stdin, b"Q")
            except Exception:
                pass
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        read_fd = getattr(self, "_read_fd", -1)
        if read_fd >= 0:
            os.close(read_fd)
            self._read_fd = -1

    def __enter__(self) -> "SonicTrtInferenceBridge":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
