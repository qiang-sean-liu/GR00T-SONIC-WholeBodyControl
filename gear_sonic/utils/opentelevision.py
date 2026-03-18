# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Originally from LW-BenchHub (lw_benchhub/utils/opentelevision.py).
# Copied here so that quest_zmq_publisher.py can run standalone without
# an LW-BenchHub checkout.

import asyncio
import traceback
from multiprocessing import Array, Process, Value, shared_memory

import numpy as np

# Monkey-patch aiohttp's BaseProtocol.resume_writing to work around an
# AssertionError ("assert self._paused") in aiohttp >=3.13 with Python 3.10
# SSL transports.  The assertion fires when the SSL transport calls
# resume_writing before the protocol has been paused, which is harmless
# but crashes the WebSocket connection.
try:
    import aiohttp.base_protocol as _abp
    _orig_resume_writing = _abp.BaseProtocol.resume_writing
    def _safe_resume_writing(self):
        if not getattr(self, "_paused", False):
            return
        _orig_resume_writing(self)
    _abp.BaseProtocol.resume_writing = _safe_resume_writing
except Exception:
    pass

from vuer import Vuer
from vuer.schemas import DefaultScene, Hands, Head, ImageBackground, MotionControllers, WebRTCStereoVideoPlane


class OpenTeleVision:
    def __init__(self, img_shape, shm_name, device_type, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=True):
        # device_type: "controller" or "hand"
        self.device_type = device_type
        self.img_shape = (img_shape[0], img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]
        self.img_width = self.img_width // 2

        self.shm_name = shm_name
        self.stream_mode = stream_mode
        self.cert_file = cert_file
        self.key_file = key_file
        self.ngrok = ngrok

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        self.left_controller_state_shared = Array('d', 7, lock=True)
        self.right_controller_state_shared = Array('d', 7, lock=True)

        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    def run(self):
        import sys
        _log = open("/tmp/opentelevision_subprocess.log", "w", buffering=1)
        sys.stdout = _log
        sys.stderr = _log
        try:
            self._run_inner()
        except Exception:
            traceback.print_exc()
            raise

    def _run_inner(self):
        print(f"[OpenTeleVision] subprocess started, device_type={self.device_type}, ngrok={self.ngrok}", flush=True)
        if self.ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
            # EnvVar descriptors cache None at import time; write directly to the
            # instance __dict__ to bypass the descriptor on subsequent reads.
            self.app.__dict__["cert"] = self.cert_file
            self.app.__dict__["key"] = self.key_file
            print(f"[OpenTeleVision] TLS cert={self.app.cert}  key={self.app.key}", flush=True)

        if self.device_type == "hand":
            self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        elif self.device_type == "controller":
            self.app.add_handler("CONTROLLER_MOVE")(self.on_motion_controller_move)
        else:
            raise ValueError("device_type must be either 'hand' or 'controller'")

        # HEAD_MOVE fires in VR mode via XRFrame.getViewerPose (the Head component).
        # CAMERA_MOVE fires in flat/desktop mode via OrbitControls.
        # Register both so head tracking works in either mode.
        self.app.add_handler("HEAD_MOVE")(self.on_head_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)

        if self.stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=self.shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
            self.app.spawn(start=True)(self.main_image)
        elif self.stream_mode == "webrtc":
            self.app.spawn(start=True)(self.main_webrtc)
        else:
            raise ValueError("stream_mode must be either 'webrtc' or 'image'")

    def close(self):
        print("closing tv")
        self.process.kill()

    async def on_head_move(self, event, session, fps=60):
        """HEAD_MOVE fires in VR mode from the Head component (XRFrame.getViewerPose)."""
        try:
            matrix = event.value.get("matrix") if isinstance(event.value, dict) else None
            if matrix is not None and isinstance(matrix, list) and len(matrix) == 16:
                self.head_matrix_shared[:] = matrix
                print(f"[on_head_move] got head matrix, pos=({matrix[12]:.3f}, {matrix[13]:.3f}, {matrix[14]:.3f})")
        except Exception as e:
            print(f"on_head_move error: {e}")

    async def on_cam_move(self, event, session, fps=60):
        """CAMERA_MOVE fires in flat/desktop mode from OrbitControls."""
        try:
            if isinstance(event.value, dict) and "camera" in event.value:
                self.head_matrix_shared[:] = event.value["camera"]["matrix"]
                self.aspect_shared.value = event.value['camera']['aspect']
            elif isinstance(event.value, dict) and "matrix" in event.value:
                self.head_matrix_shared[:] = event.value["matrix"]
        except Exception as e:
            print(f"on_cam_move error: {e}")

    async def on_motion_controller_move(self, event, session, fps=60):
        if "right" in event.value:
            self._parse_hand_data(event.value, "right", self.right_hand_shared)
            self._parse_controller_state(event.value, "right", self.right_controller_state_shared)
        if "left" in event.value:
            self._parse_hand_data(event.value, "left", self.left_hand_shared)
            self._parse_controller_state(event.value, "left", self.left_controller_state_shared)

    def _parse_controller_state(self, value, side, controller_state_shared):
        value = value[f"{side}State"]
        if len(value) == 0:
            return
        values = [
            float(value["triggerValue"]),      # 0 to 1
            float(value["squeezeValue"]),       # 0 to 1
            -float(value["thumbstickValue"][1]),# -1 to 1
            -float(value["thumbstickValue"][0]),# -1 to 1
            float(value["thumbstick"]),         # 0 or 1
            float(value["aButton"]),            # 0 or 1
            float(value["bButton"]),            # 0 or 1
        ]
        controller_state_shared[:] = np.array(values)

    def _parse_hand_data(self, value, side, hand_shared, landmarks_shared=None):
        if side not in value:
            return
        data = value[side]
        # vuer >= 0.1.5: Float32Array arrives as msgpack ExtType (raw bytes).
        # Older vuer: arrives as a Python list of floats.
        if hasattr(data, 'data'):  # msgpack.ExtType
            raw = data.data
            if len(raw) < 64:  # need at least 16 floats (4x4 wrist matrix) = 64 bytes
                return
            data = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
        elif isinstance(data, list):
            data = np.array(data)
        else:
            return
        if len(data) < 16:
            return
        hand_shared[:] = data[:16]
        if landmarks_shared is not None and len(data) >= 400:
            data[:400].reshape(25, 4, 4).transpose(0, 2, 1)
            landmarks_shared[:] = data[:400].reshape(25, 4, 4).transpose(0, 2, 1)[:, :3, 3].flatten()

    _hand_log_counter = 0

    async def on_hand_move(self, event, session, fps=60):
        self._hand_log_counter = getattr(self, '_hand_log_counter', 0) + 1
        if self._hand_log_counter <= 5 or self._hand_log_counter % 300 == 0:
            all_keys = list(event.__dict__.keys())
            val_keys = list(event.value.keys()) if isinstance(event.value, dict) else type(event.value).__name__
            print(f"[on_hand_move #{self._hand_log_counter}] event_keys={all_keys} value_keys={val_keys}")
            if self._hand_log_counter <= 3:
                # Print raw event dict so we can see where data actually lives
                print(f"  raw event.__dict__={str(event.__dict__)[:600]}")

        try:
            self._parse_hand_data(event.value, "left", self.left_hand_shared, self.left_landmarks_shared)
        except Exception as e:
            traceback.print_exc()
            print(f"on left hand move error: {e}")
        try:
            self._parse_hand_data(event.value, "right", self.right_hand_shared, self.right_landmarks_shared)
        except Exception as e:
            traceback.print_exc()
            print(f"on right hand move error: {e}")

    async def main_webrtc(self, session, fps=60):
        session.set @ DefaultScene(frameloop="always")
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        session.upsert @ Head(stream=True, fps=fps, key="head_tracking")
        session.upsert @ WebRTCStereoVideoPlane(
            src="https://192.168.8.102:8080/offer",
            key="zed",
            aspect=1.33334,
            height=8,
            position=[0, -2, -0.2],
        )
        while True:
            await asyncio.sleep(1)

    async def main_image(self, session, fps=60):
        print(f"[main_image] session started, device_type={self.device_type}", flush=True)
        session.set @ DefaultScene(frameloop="always")
        # Head component uses XRFrame.getViewerPose -> HEAD_MOVE events (works in VR mode).
        session.upsert @ Head(stream=True, fps=fps, key="head_tracking")
        if self.device_type == "hand":
            session.upsert @ Hands(fps=fps, stream=True, key="hands")
        elif self.device_type == "controller":
            session.upsert @ MotionControllers(stream=True, key="motion-controller-right", right=True)
            # NOTE: Two controllers simultaneously is only supported on PICO.
            # On Quest, adding a second MotionControllers element freezes the browser.
            session.upsert @ MotionControllers(stream=True, key="motion-controller-left", left=True)
        print("[main_image] scene components sent. Waiting for Quest to enter VR mode...")

        _frame_interval = 1.0 / 30.0  # 30 Hz image stream
        _encode_errors = 0
        while True:
            t0 = asyncio.get_event_loop().time()
            try:
                frame = self.img_array.copy()
                if np.any(frame):  # skip blank (all-zero) frames until sim starts rendering
                    # Pass numpy array directly; vuer encodes it as JPEG internally.
                    session.upsert @ ImageBackground(
                        frame,
                        format="jpeg",
                        quality=75,
                        key="head_cam",
                    )
                    _encode_errors = 0
            except Exception as e:
                _encode_errors += 1
                if _encode_errors <= 3 or _encode_errors % 100 == 0:
                    print(f"[main_image] frame encode error #{_encode_errors}: {e}", flush=True)

            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0.0, _frame_interval - elapsed))

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)

    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def left_controller_state(self):
        trigger, squeeze, thumbstick_x, thumbstick_y, thumbstick, a_button, b_button = self.left_controller_state_shared
        return {
            "trigger": trigger,
            "squeeze": squeeze,
            "thumbstick_x": thumbstick_x,
            "thumbstick_y": thumbstick_y,
            "thumbstick": bool(thumbstick),
            "a_button": bool(a_button),
            "b_button": bool(b_button),
        }

    @property
    def right_controller_state(self):
        trigger, squeeze, thumbstick_x, thumbstick_y, thumbstick, a_button, b_button = self.right_controller_state_shared
        return {
            "trigger": trigger,
            "squeeze": squeeze,
            "thumbstick_x": thumbstick_x,
            "thumbstick_y": thumbstick_y,
            "thumbstick": bool(thumbstick),
            "a_button": bool(a_button),
            "b_button": bool(b_button),
        }

    @property
    def aspect(self):
        return float(self.aspect_shared.value)
