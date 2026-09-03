# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton._triton_kernels.rope.fused_qkv_split_qk_norm_rope_cat import (
    _fused_qkv_split_qk_norm_rope_cat_kernel,
)
from aiter.ops.triton.utils.device_info import get_num_sms

_NUM_HEADS = 24
_HEAD_DIM = 128
_PACKED_DIM = 3 * _NUM_HEADS * _HEAD_DIM


def _validate_packed_qkv(name: str, tensor: torch.Tensor) -> int:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on a CUDA/HIP device")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use torch.bfloat16")
    if tensor.ndim not in (2, 3):
        raise ValueError(
            f"{name} must have shape [tokens, {_PACKED_DIM}] "
            f"or [1, tokens, {_PACKED_DIM}]"
        )
    if tensor.ndim == 3 and tensor.shape[0] != 1:
        raise ValueError(f"{name} only supports batch size one")
    if tensor.shape[-1] != _PACKED_DIM:
        raise ValueError(f"{name} must have packed dimension {_PACKED_DIM}")
    if tensor.shape[-2] == 0:
        raise ValueError(f"{name} must contain at least one token")
    if tensor.stride(-1) != 1 or tensor.stride(-2) != _PACKED_DIM:
        raise ValueError(f"{name} must use the contiguous packed QKV token layout")
    return tensor.shape[-2]


def _validate_vector(
    name: str,
    tensor: torch.Tensor,
    like: torch.Tensor,
) -> None:
    if tensor.device != like.device:
        raise ValueError(f"{name} must be on {like.device}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use torch.bfloat16")
    if tensor.shape != (_HEAD_DIM,) or tensor.stride(0) != 1:
        raise ValueError(f"{name} must be a contiguous [{_HEAD_DIM}] tensor")


def _validate_rope_table(
    name: str,
    tensor: torch.Tensor,
    like: torch.Tensor,
    tokens: int,
) -> None:
    if tensor.device != like.device:
        raise ValueError(f"{name} must be on {like.device}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use torch.bfloat16")
    if tensor.ndim != 2 or tensor.shape != (tokens, _HEAD_DIM):
        raise ValueError(f"{name} must have shape [{tokens}, {_HEAD_DIM}]")
    if tensor.stride(1) != 1 or tensor.stride(0) != _HEAD_DIM:
        raise ValueError(f"{name} must be contiguous")


def _block_tokens(total_tokens: int) -> int:
    compute_units = max(int(get_num_sms()), 1)
    block_tokens = triton.next_power_of_2(triton.cdiv(total_tokens, 2 * compute_units))
    # Larger token tiles change Triton's RMSNorm reduction layout and no
    # longer reproduce PyTorch's BF16 result exactly on gfx942.
    return max(1, min(int(block_tokens), 2))


def fused_qkv_split_qk_norm_rope_cat(
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    img_rope: torch.Tensor,
    txt_rope: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dense text-first QKV after Q/K RMSNorm and interleaved RoPE.

    ``img_qkv`` and ``txt_qkv`` contain packed Q, K, and V projections. Each
    RoPE table stores ``[cos, sin]`` in its final dimension. This first version
    supports only the BF16 Qwen Image layout used on ``gfx942``.
    """
    img_tokens = _validate_packed_qkv("img_qkv", img_qkv)
    txt_tokens = _validate_packed_qkv("txt_qkv", txt_qkv)
    if txt_qkv.device != img_qkv.device:
        raise ValueError("img_qkv and txt_qkv must be on the same device")

    architecture = get_gfx_runtime()
    if architecture != "gfx942":
        raise ValueError(
            f"fused operation requires gfx942, got {architecture or 'unknown'}"
        )

    for name, tensor in (
        ("img_q_weight", img_q_weight),
        ("img_k_weight", img_k_weight),
        ("txt_q_weight", txt_q_weight),
        ("txt_k_weight", txt_k_weight),
    ):
        _validate_vector(name, tensor, img_qkv)
    _validate_rope_table("img_rope", img_rope, img_qkv, img_tokens)
    _validate_rope_table("txt_rope", txt_rope, img_qkv, txt_tokens)

    total_tokens = img_tokens + txt_tokens
    output_shape = (total_tokens, _NUM_HEADS, _HEAD_DIM)
    joint_q = torch.empty(output_shape, dtype=img_qkv.dtype, device=img_qkv.device)
    joint_k = torch.empty_like(joint_q)
    joint_v = torch.empty_like(joint_q)
    block_tokens = _block_tokens(total_tokens)
    grid = (triton.cdiv(total_tokens, block_tokens), _NUM_HEADS)

    _fused_qkv_split_qk_norm_rope_cat_kernel[grid](
        img_qkv,
        txt_qkv,
        img_q_weight,
        img_k_weight,
        txt_q_weight,
        txt_k_weight,
        img_rope,
        txt_rope,
        joint_q,
        joint_k,
        joint_v,
        img_tokens,
        txt_tokens,
        eps,
        img_qkv.stride(-2),
        txt_qkv.stride(-2),
        img_rope.stride(0),
        txt_rope.stride(0),
        joint_q.stride(0),
        joint_q.stride(1),
        NUM_HEADS=_NUM_HEADS,
        BLOCK_T=block_tokens,
        BLOCK_D=_HEAD_DIM,
        BLOCK_D_HALF=_HEAD_DIM // 2,
        num_warps=4,
    )
    return joint_q, joint_k, joint_v

