# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""`camera_layout="robotwin_tshape"`: the frame composition the RoboTwin FastWAM checkpoints were
trained with (head camera on top, two wrist cameras side by side underneath)."""

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import _resize_frames, _stack_video_from_images
from lerobot.utils.constants import ACTION, OBS_STATE

DATASET_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ENV_KEYS = (
    "observation.images.head_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
)


def _config(camera_layout: str | None = None, raw_cameras: bool = True) -> FastWAMConfig:
    """Config for a 384x320 frame. With `raw_cameras` the three RoboTwin cameras are declared at their
    native 480x640, which only the T-shape layout accepts (it resizes them itself)."""
    features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))}
    if raw_cameras:
        features.update(
            {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for k in DATASET_KEYS}
        )
    else:  # three strips that already satisfy the default layout's width check
        widths = (106, 106, 108)
        features.update(
            {
                k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 384, w))
                for k, w in zip(DATASET_KEYS, widths, strict=True)
            }
        )
    return FastWAMConfig(
        device="cpu",
        image_size=(384, 320),
        action_dim=14,
        proprio_dim=14,
        input_features=features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        camera_layout=camera_layout,
    )


def _cams(keys, shape=(1, 3, 480, 640)):
    torch.manual_seed(0)
    return {k: torch.rand(*shape) for k in keys}


def test_default_layout_concatenates_along_width():
    video = _stack_video_from_images(_cams(DATASET_KEYS), _config(raw_cameras=False))
    # three 384x106 strips: 318 wide, later resized to 320 by the model
    assert tuple(video.shape[-2:]) == (384, 318)


@pytest.mark.parametrize("keys", [DATASET_KEYS, ENV_KEYS])
def test_tshape_composes_head_over_wrists(keys):
    cams = _cams(keys)
    video = _stack_video_from_images(cams, _config(camera_layout="robotwin_tshape"))
    assert tuple(video.shape) == (1, 3, 9, 384, 320)
    frame = video[:, :, 0]
    head, left, right = (cams[k] for k in keys)
    torch.testing.assert_close(frame[..., :256, :], _resize_frames(head, (256, 320)))
    torch.testing.assert_close(frame[..., 256:, :160], _resize_frames(left, (128, 160)))
    torch.testing.assert_close(frame[..., 256:, 160:], _resize_frames(right, (128, 160)))


def test_tshape_handles_temporal_stacks():
    cams = _cams(DATASET_KEYS, shape=(1, 3, 3, 480, 640))  # [B, T, C, H, W]
    video = _stack_video_from_images(cams, _config(camera_layout="robotwin_tshape"))
    assert tuple(video.shape) == (1, 3, 3, 384, 320)


def test_tshape_accepts_precomposed_single_image():
    batch = {"observation.images.image": torch.rand(1, 3, 384, 320)}
    video = _stack_video_from_images(batch, _config(camera_layout="robotwin_tshape"))
    assert tuple(video.shape) == (1, 3, 9, 384, 320)


def test_tshape_requires_all_three_cameras():
    cams = _cams(DATASET_KEYS[:2])
    with pytest.raises(KeyError, match="right camera"):
        _stack_video_from_images(cams, _config(camera_layout="robotwin_tshape"))


def test_config_validation_skips_width_check_under_a_layout():
    # Default layout rejects raw 480x640 cameras against a 384x320 frame; the T-shape resizes them itself.
    with pytest.raises(ValueError, match="height must be"):
        _config()
    _config(camera_layout="robotwin_tshape")
    with pytest.raises(ValueError, match="Unsupported camera_layout"):
        _config(camera_layout="grid")
