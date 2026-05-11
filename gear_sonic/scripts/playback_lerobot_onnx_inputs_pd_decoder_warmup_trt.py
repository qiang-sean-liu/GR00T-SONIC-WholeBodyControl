"""Warmup playback that keeps Python MuJoCo logic but runs SONIC inference in C++ TensorRT."""

from __future__ import annotations

import argparse

import playback_lerobot_onnx_inputs_pd_decoder_warmup as warmup
from playback_lerobot import SonicRunner as _BaseSonicRunner
from sonic_trt_inference_bridge import SonicTrtInferenceBridge


class TrtSonicRunner(_BaseSonicRunner):
    def __init__(self, encoder_path: str, decoder_path: str, fps: float = 20.0, closed_loop: bool = True):
        self._trt_bridge = SonicTrtInferenceBridge(
            encoder_path,
            decoder_path,
            executable=TrtSonicRunner.executable,
            encoder_fp16=TrtSonicRunner.encoder_fp16,
            decoder_fp16=TrtSonicRunner.decoder_fp16,
        )
        super().__init__(
            encoder_path,
            decoder_path,
            fps=fps,
            closed_loop=closed_loop,
            inference_backend=self._trt_bridge,
        )

    def close(self) -> None:
        self._trt_bridge.close()


TrtSonicRunner.executable = "gear_sonic_deploy/target/release/sonic_trt_inference_bridge"
TrtSonicRunner.encoder_fp16 = False
TrtSonicRunner.decoder_fp16 = False


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
    parser.add_argument("--init_substep_frame", type=int, default=None)
    parser.add_argument("--init_substep_index", type=int, default=0)
    parser.add_argument("--init_substep_all_inputs", action="store_true")
    parser.add_argument(
        "--trt_executable",
        default=TrtSonicRunner.executable,
        help="Path to the built C++ TensorRT inference bridge executable.",
    )
    parser.add_argument("--encoder_fp16", action="store_true")
    parser.add_argument("--decoder_fp16", action="store_true")

    args = parser.parse_args()
    TrtSonicRunner.executable = args.trt_executable
    TrtSonicRunner.encoder_fp16 = args.encoder_fp16
    TrtSonicRunner.decoder_fp16 = args.decoder_fp16

    original_runner = warmup.SonicRunner
    warmup.SonicRunner = TrtSonicRunner
    try:
        warmup.playback(args)
    finally:
        warmup.SonicRunner = original_runner


if __name__ == "__main__":
    main()
