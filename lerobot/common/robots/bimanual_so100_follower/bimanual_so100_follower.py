#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_bimanual_so100_follower import BimanualSO100FollowerConfig

from ..so100_follower import SO100Follower

logger = logging.getLogger(__name__)


class BimanualSO100Follower(Robot):
    """
    [SO-100 Follower Arm](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = BimanualSO100FollowerConfig
    name = "so100_follower"

    def __init__(self, config: BimanualSO100FollowerConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = SO100Follower(config.left_arm)
        self.right_arm = SO100Follower(config.right_arm)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_motors_ft = self.left_arm._motors_ft
        right_motors_ft = self.right_arm._motors_ft
        combined_motors_ft = {}
        for key in left_motors_ft:
            combined_motors_ft[f"left_{key}"] = left_motors_ft[key]
        for key in right_motors_ft:
            combined_motors_ft[f"right_{key}"] = right_motors_ft[key]
        return combined_motors_ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        self.left_arm.connect(calibrate=calibrate)
        self.right_arm.connect(calibrate=calibrate)
        for cam in self.cameras.values():
            cam.connect()
        
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        raise NotImplementedError("Calibration for BimanualSO100Follower is not implemented.")

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()
        logger.info(f"{self} configured.")

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()
        logger.info(f"{self} motors setup complete.")

    def get_observation(self) -> dict[str, Any]:
        left_obs = self.left_arm.get_observation()
        right_obs = self.right_arm.get_observation()
        combined_obs = {}
        for key in left_obs:
            combined_obs[f"left_{key}"] = left_obs[key]
        for key in right_obs:
            combined_obs[f"right_{key}"] = right_obs[key]

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            combined_obs[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return combined_obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_action = {key.removeprefix("left_"): val for key, val in action.items() if key.startswith("left_")}
        right_action = {key.removeprefix("right_"): val for key, val in action.items() if key.startswith("right_")}

        left_pos = self.left_arm.send_action(left_action)
        right_pos = self.right_arm.send_action(right_action)

        combined_action = {}
        for key in left_pos:
            combined_action[f"left_{key}"] = left_pos[key]
        for key in right_pos:
            combined_action[f"right_{key}"] = right_pos[key]
        
        logger.debug(f"{self} sent action: {combined_action}")

        return combined_action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
