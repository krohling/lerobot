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

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_bimanual_so100_leader import BimanualSO100LeaderConfig

from ..so100_leader import SO100Leader

logger = logging.getLogger(__name__)


class BimanualSO100Leader(Teleoperator):
    """
    [SO-100 Leader Arm](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = BimanualSO100LeaderConfig
    name = "bimanual_so100_leader"

    def __init__(self, config: BimanualSO100LeaderConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = SO100Leader(config.left_arm)
        self.right_arm = SO100Leader(config.right_arm)

    @property
    def action_features(self) -> dict[str, type]:
        left_action_features = self.left_arm.action_features
        right_action_features = self.right_arm.action_features
        combined_action_features = {}
        for key in left_action_features:
            combined_action_features[f"left_{key}"] = left_action_features[key]
        for key in right_action_features:
            combined_action_features[f"right_{key}"] = right_action_features[key]
        return combined_action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        self.left_arm.connect(calibrate=calibrate)
        self.right_arm.connect(calibrate=calibrate)

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated
    
    def calibrate(self) -> None:
        raise NotImplementedError("Calibration for BimanualSO100Leader is not implemented.")


    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()
        logger.info(f"{self} configured.")

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()
        logger.info(f"{self} motors setup complete.")

    def get_action(self) -> dict[str, float]:
        left_action = self.left_arm.get_action()
        right_action = self.right_arm.get_action()
        combined_action = {}
        for key in left_action:
            combined_action[f"left_{key}"] = left_action[key]
        for key in right_action:
            combined_action[f"right_{key}"] = right_action[key]
        logger.debug(f"{self} combined action: {combined_action}")
        return combined_action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.left_arm.disconnect()
        self.right_arm.disconnect()
        logger.info(f"{self} disconnected.")
