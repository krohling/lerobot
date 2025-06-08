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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

from pathlib import Path

@dataclass(kw_only=True)
class BimanualSO100LeaderArmConfig():
    # Allows to distinguish between different teleoperators of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None
    # Port to connect to the arm
    port: str

@TeleoperatorConfig.register_subclass("bimanual_so100_leader")
@dataclass
class BimanualSO100LeaderConfig(TeleoperatorConfig):
    left_arm: BimanualSO100LeaderArmConfig = field(default_factory=lambda: BimanualSO100LeaderArmConfig(port="/dev/ttyUSB0"))
    right_arm: BimanualSO100LeaderArmConfig = field(default_factory=lambda: BimanualSO100LeaderArmConfig(port="/dev/ttyUSB1"))
