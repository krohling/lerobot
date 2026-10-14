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
"""FastWAM on devices without float64 or CUDA autocast (MPS).

The Wan backbone computes its sinusoidal and rotary embeddings in float64 and keeps the time
embedding and output head in float32 through a CUDA-only autocast region. Both fail on MPS: the
first raises, the second aborts the process inside Metal on a mixed-dtype matmul. These tests pin
the device-independent helpers on CPU and, when MPS is available, check that the MPS path matches
the CPU result.
"""

import pytest
import torch
from torch import nn

from lerobot.policies.fastwam.wan.model import (
    Head,
    fp32_call,
    move_freqs,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")


def test_sinusoidal_embedding_keeps_float64_on_cpu():
    out = sinusoidal_embedding_1d(16, torch.arange(4))
    assert out.dtype == torch.float64 and out.shape == (4, 16)


def test_move_freqs_is_identity_on_cpu():
    freqs = rope_params(8, 4)
    assert freqs.dtype == torch.complex128
    moved = move_freqs(freqs, "cpu")
    assert moved.dtype == torch.complex128 and moved.device.type == "cpu"


def test_fp32_call_runs_bf16_module_in_float32_on_cpu():
    torch.manual_seed(0)
    lin = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4)).to(torch.bfloat16)
    x = torch.randn(3, 8)
    out = fp32_call(lin, x)
    assert out.dtype == torch.float32
    reference = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))
    reference.load_state_dict({k: v.float() for k, v in lin.state_dict().items()})
    torch.testing.assert_close(out, reference(x))
    # The module's own parameters are untouched.
    assert all(p.dtype == torch.bfloat16 for p in lin.parameters())


@MPS
def test_sinusoidal_embedding_matches_cpu_on_mps():
    position = torch.arange(6)
    cpu = sinusoidal_embedding_1d(32, position)
    mps = sinusoidal_embedding_1d(32, position.to("mps"))
    assert mps.dtype == torch.float32
    torch.testing.assert_close(mps.cpu(), cpu.float(), atol=1e-5, rtol=1e-5)


@MPS
def test_move_freqs_downcasts_to_complex64_on_mps():
    freqs = rope_params(8, 4)
    moved = move_freqs(freqs, "mps")
    assert moved.dtype == torch.complex64 and moved.device.type == "mps"
    torch.testing.assert_close(moved.cpu(), freqs.to(torch.complex64))


@MPS
def test_rope_apply_matches_cpu_on_mps():
    torch.manual_seed(0)
    f, h, w, n, d = 2, 3, 4, 2, 12  # head dim 12: 4 + 4 + 4 split
    x = torch.randn(1, f * h * w, n, d)
    grid = torch.tensor([[f, h, w]])
    freqs = torch.cat(
        [rope_params(16, d - 4 * (d // 6)), rope_params(16, 2 * (d // 6)), rope_params(16, 2 * (d // 6))],
        dim=1,
    )
    cpu = rope_apply(x, grid, freqs)
    mps = rope_apply(x.to("mps"), grid, freqs)
    torch.testing.assert_close(mps.cpu(), cpu, atol=1e-4, rtol=1e-4)


@MPS
def test_head_forward_matches_cpu_on_mps_in_bf16():
    torch.manual_seed(0)
    head = Head(dim=16, out_dim=4, patch_size=(1, 2, 2), eps=1e-6).to(torch.bfloat16)
    x = torch.randn(1, 5, 16).to(torch.bfloat16)
    e = torch.randn(1, 5, 16).to(torch.bfloat16)
    cpu = head(x, e)
    head_mps = Head(dim=16, out_dim=4, patch_size=(1, 2, 2), eps=1e-6).to(torch.bfloat16)
    head_mps.load_state_dict(head.state_dict())
    mps = head_mps.to("mps")(x.to("mps"), e.to("mps"))
    torch.testing.assert_close(mps.cpu().float(), cpu.float(), atol=2e-2, rtol=2e-2)
