# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cat as fused_qkv_module
import pytest
import torch
import torch.nn.functional as F
from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cat import (
    fused_qkv_split_qk_norm_rope_cat,
)

_NUM_HEADS = 24
_HEAD_DIM = 128
_PACKED_DIM = 3 * _NUM_HEADS * _HEAD_DIM
_EPS = 1e-6


def _rope_table(tokens: int) -> torch.Tensor:
    angles = torch.randn((tokens, _HEAD_DIM // 2), device="cuda")
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)


def _apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    x_f32 = x.float().reshape(x.shape[0], _NUM_HEADS, _HEAD_DIM // 2, 2)
    cos = rope[:, : _HEAD_DIM // 2].float()[:, None, :]
    sin = rope[:, _HEAD_DIM // 2 :].float()[:, None, :]
    first = x_f32[..., 0]
    second = x_f32[..., 1]
    result = torch.stack(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )
    return result.reshape_as(x).to(torch.bfloat16)


def _stream_reference(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = qkv.split(_NUM_HEADS * _HEAD_DIM, dim=-1)
    q = q.view(-1, _NUM_HEADS, _HEAD_DIM)
    k = k.view(-1, _NUM_HEADS, _HEAD_DIM)
    v = v.view(-1, _NUM_HEADS, _HEAD_DIM)
    q = F.rms_norm(q, (_HEAD_DIM,), q_weight, _EPS)
    k = F.rms_norm(k, (_HEAD_DIM,), k_weight, _EPS)
    return _apply_rope(q, rope), _apply_rope(k, rope), v


def _reference(
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    img_rope: torch.Tensor,
    txt_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img = _stream_reference(img_qkv, weights[0], weights[1], img_rope)
    txt = _stream_reference(txt_qkv, weights[2], weights[3], txt_rope)
    return tuple(torch.cat((txt[i], img[i]), dim=0) for i in range(3))


def _small_inputs() -> tuple[torch.Tensor, ...]:
    img_qkv = torch.empty((1, 2, _PACKED_DIM), device="cuda", dtype=torch.bfloat16)
    txt_qkv = torch.empty((1, 2, _PACKED_DIM), device="cuda", dtype=torch.bfloat16)
    weights = tuple(
        torch.empty((_HEAD_DIM,), device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    rope = torch.empty((2, _HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    return img_qkv, txt_qkv, *weights, rope, rope


@pytest.mark.parametrize("input_rank", [2, 3])
@pytest.mark.parametrize("img_tokens,txt_tokens", [(1024, 7), (4096, 13), (4096, 5)])
def test_fused_qkv_split_qk_norm_rope_cat(
    img_tokens: int,
    txt_tokens: int,
    input_rank: int,
):
    torch.manual_seed(1)
    img_qkv = torch.randn(
        (1, img_tokens, _PACKED_DIM), device="cuda", dtype=torch.bfloat16
    )
    txt_qkv = torch.randn(
        (1, txt_tokens, _PACKED_DIM), device="cuda", dtype=torch.bfloat16
    )
    if input_rank == 2:
        img_qkv = img_qkv.squeeze(0)
        txt_qkv = txt_qkv.squeeze(0)
    weights = tuple(
        torch.randn((_HEAD_DIM,), device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    img_rope = _rope_table(img_tokens)
    txt_rope = _rope_table(txt_tokens)

    img_q, img_k, img_v = img_qkv.split(_NUM_HEADS * _HEAD_DIM, dim=-1)
    assert img_q.stride() == img_k.stride() == img_v.stride()
    assert img_q.stride(-2) == _PACKED_DIM
    assert not img_q.is_contiguous()

    expected = _reference(img_qkv, txt_qkv, weights, img_rope, txt_rope)
    actual = fused_qkv_split_qk_norm_rope_cat(
        img_qkv, txt_qkv, *weights, img_rope, txt_rope, _EPS
    )
    repeated = fused_qkv_split_qk_norm_rope_cat(
        img_qkv, txt_qkv, *weights, img_rope, txt_rope, _EPS
    )

    for name, expected_tensor, actual_tensor, repeated_tensor in zip(
        ("q", "k", "v"), expected, actual, repeated
    ):
        difference = expected_tensor.float() - actual_tensor.float()
        assert torch.equal(expected_tensor, actual_tensor), (
            f"{name} mismatch: count={torch.count_nonzero(difference).item()}, "
            f"max={difference.abs().max().item()}"
        )
        assert torch.equal(actual_tensor, repeated_tensor)
        assert actual_tensor.is_contiguous()


def test_fused_qkv_split_qk_norm_rope_cat_rejects_wrong_stride():
    inputs = list(_small_inputs())
    txt_qkv = torch.empty((2, _PACKED_DIM * 2), device="cuda", dtype=torch.bfloat16)[
        :, ::2
    ]
    inputs[1] = txt_qkv

    with pytest.raises(ValueError, match="contiguous packed QKV"):
        fused_qkv_split_qk_norm_rope_cat(*inputs)


@pytest.mark.parametrize(
    "index,shape,error",
    [
        (0, (1, 0, _PACKED_DIM), "at least one token"),
        (1, (0, _PACKED_DIM), "at least one token"),
        (0, (2, 2, _PACKED_DIM), "batch size one"),
        (1, (2, 2, _PACKED_DIM), "batch size one"),
    ],
)
def test_fused_qkv_split_qk_norm_rope_cat_rejects_empty_or_batched_inputs(
    index: int,
    shape: tuple[int, ...],
    error: str,
):
    inputs = list(_small_inputs())
    inputs[index] = torch.empty(shape, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=error):
        fused_qkv_split_qk_norm_rope_cat(*inputs)


@pytest.mark.parametrize(
    "index,replacement,error",
    [
        (
            0,
            lambda: torch.empty(
                (1, 2, _PACKED_DIM), device="cuda", dtype=torch.float16
            ),
            "torch.bfloat16",
        ),
        (
            1,
            lambda: torch.empty(
                (1, 2, _PACKED_DIM - 1), device="cuda", dtype=torch.bfloat16
            ),
            "packed dimension",
        ),
        (
            2,
            lambda: torch.empty((_HEAD_DIM + 1,), device="cuda", dtype=torch.bfloat16),
            "contiguous.*tensor",
        ),
        (
            6,
            lambda: torch.empty((3, _HEAD_DIM), device="cuda", dtype=torch.bfloat16),
            "must have shape",
        ),
    ],
)
def test_fused_qkv_split_qk_norm_rope_cat_rejects_invalid_inputs(
    index: int,
    replacement,
    error: str,
):
    inputs = list(_small_inputs())
    inputs[index] = replacement()

    with pytest.raises(ValueError, match=error):
        fused_qkv_split_qk_norm_rope_cat(*inputs)


def test_fused_qkv_split_qk_norm_rope_cat_rejects_other_arch(monkeypatch):
    monkeypatch.setattr(fused_qkv_module, "get_gfx_runtime", lambda: "gfx950")

    with pytest.raises(ValueError, match="requires gfx942, got gfx950"):
        fused_qkv_split_qk_norm_rope_cat(*_small_inputs())
