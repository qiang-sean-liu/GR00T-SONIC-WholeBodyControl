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

from vuer import Vuer
from vuer.schemas import DefaultScene, Hands, MotionControllers, WebRTCStereoVideoPlane


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
        self.left_controller_state_shared = Array('d', 7, lock=True)  # trigger, squeeze, thumbstick_x, thumbstick_y, thumbstick, a_button, b_button
        self.right_controller_state_shared = Array('d', 7, lock=True)  # trigger, squeeze, thumbstick_x, thumbstick_y, thumbstick, a_button, b_button

        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    def run(self):
        if self.ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', cert=self.cert_file, key=self.key_file, queries=dict(grid=False), queue_len=3)

        if self.device_type == "hand":
            self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        elif self.device_type == "controller":
            self.app.add_handler("CONTROLLER_MOVE")(self.on_motion_controller_move)
        else:
            raise ValueError("device_type must be either 'hand' or 'controller'")
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

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except Exception as e:
            print(f"on cam move error: {e}. event.value=\n{event.value}")

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
        data = value[side]
        if not isinstance(data, list):
            # when the hand is not detected, the data is not a list.
            return
        data = np.array(data)
        hand_shared[:] = data[:16]
        if landmarks_shared is not None:
            data = data.reshape(25, 4, 4).transpose(0, 2, 1)
            landmarks_shared[:] = data[:, :3, 3].flatten()  # only use the position

    async def on_hand_move(self, event, session, fps=60):
        try:
            self._parse_hand_data(event.value, "left", self.left_hand_shared, self.left_landmarks_shared)
        except Exception as e:
            traceback.print_exc()
            print(f"on left hand move error: {e}. event.value=\n{event.value}")
        try:
            self._parse_hand_data(event.value, "right", self.right_hand_shared, self.right_landmarks_shared)
        except Exception as e:
            traceback.print_exc()
            print(f"on right hand move error: {e}. event.value=\n{event.value}")

    async def main_webrtc(self, session, fps=60):
        session.set @ DefaultScene(frameloop="always")
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
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
        if self.device_type == "hand":
            session.upsert @ Hands(fps=fps, stream=True, key="hands")
        elif self.device_type == "controller":
            session.upsert @ MotionControllers(stream=True, key="motion-controller-right", right=True)
            # NOTE: Two controllers simultaneously is only supported on PICO.
            # On Quest, adding a second MotionControllers element freezes the browser.
            session.upsert @ MotionControllers(stream=True, key="motion-controller-left", left=True)
        while True:
            await asyncio.sleep(0.03)

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
