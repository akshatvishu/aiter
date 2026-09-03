# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from aiter.ops.triton.rope.rope import _get_gptj_rotated_x
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _load_stream(
    img_ptr,
    txt_ptr,
    img_offsets,
    txt_offsets,
    img_mask,
    txt_mask,
):
    img = tl.load(img_ptr + img_offsets, mask=img_mask, other=0.0)
    txt = tl.load(txt_ptr + txt_offsets, mask=txt_mask, other=0.0)
    return img + txt


@triton.jit
def _load_and_norm_stream(
    img_ptr,
    txt_ptr,
    img_base_offsets,
    txt_base_offsets,
    img_mask,
    txt_mask,
    weight,
    eps,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # PyTorch's vectorized RMSNorm loads groups of four contiguous values and
    # accumulates each group serially before its wave reduction. Preserve that
    # order because the BF16 result is Qwen's compatibility boundary for RoPE.
    group_offsets = 4 * tl.arange(0, BLOCK_D // 4)
    img_offsets = img_base_offsets + group_offsets[None, :]
    txt_offsets = txt_base_offsets + group_offsets[None, :]
    x0 = _load_stream(
        img_ptr, txt_ptr, img_offsets, txt_offsets, img_mask, txt_mask
    ).to(tl.float32)
    x1 = _load_stream(
        img_ptr, txt_ptr, img_offsets + 1, txt_offsets + 1, img_mask, txt_mask
    ).to(tl.float32)
    x2 = _load_stream(
        img_ptr, txt_ptr, img_offsets + 2, txt_offsets + 2, img_mask, txt_mask
    ).to(tl.float32)
    x3 = _load_stream(
        img_ptr, txt_ptr, img_offsets + 3, txt_offsets + 3, img_mask, txt_mask
    ).to(tl.float32)

    partial = x0 * x0
    partial += x1 * x1
    partial += x2 * x2
    partial += x3 * x3
    variance = tl.sum(partial, axis=1) / BLOCK_D
    inv_rms = tl.rsqrt(variance + eps)[:, None]
    x = tl.reshape(
        tl.join(tl.join(x0, x2), tl.join(x1, x3)),
        (BLOCK_T, BLOCK_D),
    )
    return (x * inv_rms * weight.to(tl.float32)).to(tl.bfloat16)


_kernel_repr = make_kernel_repr(
    "_fused_qkv_split_qk_norm_rope_cat_kernel",
    ["NUM_HEADS", "BLOCK_T", "BLOCK_D", "BLOCK_D_HALF"],
)


@triton.jit(repr=_kernel_repr)
def _fused_qkv_split_qk_norm_rope_cat_kernel(
    img_qkv_ptr,
    txt_qkv_ptr,
    img_q_weight_ptr,
    img_k_weight_ptr,
    txt_q_weight_ptr,
    txt_k_weight_ptr,
    img_rope_ptr,
    txt_rope_ptr,
    joint_q_ptr,
    joint_k_ptr,
    joint_v_ptr,
    img_tokens,
    txt_tokens,
    eps,
    img_qkv_stride_t,
    txt_qkv_stride_t,
    img_rope_stride_t,
    txt_rope_stride_t,
    joint_stride_t,
    joint_stride_h,
    NUM_HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_D)
    total_tokens = txt_tokens + img_tokens

    token_mask = token_offsets < total_tokens
    is_text = token_offsets < txt_tokens
    txt_token_offsets = token_offsets
    img_token_offsets = token_offsets - txt_tokens
    txt_mask = token_mask[:, None] & is_text[:, None]
    img_mask = token_mask[:, None] & ~is_text[:, None]

    head_offset = head * BLOCK_D
    txt_head_offsets = txt_token_offsets[:, None] * txt_qkv_stride_t + head_offset
    img_head_offsets = img_token_offsets[:, None] * img_qkv_stride_t + head_offset
    txt_base_offsets = txt_head_offsets + dim_offsets[None, :]
    img_base_offsets = img_head_offsets + dim_offsets[None, :]

    weight_offsets = dim_offsets[None, :]
    img_q_weight = tl.load(img_q_weight_ptr + weight_offsets)
    txt_q_weight = tl.load(txt_q_weight_ptr + weight_offsets)
    q_weight = tl.where(is_text[:, None], txt_q_weight, img_q_weight)

    q = _load_and_norm_stream(
        img_qkv_ptr,
        txt_qkv_ptr,
        img_head_offsets,
        txt_head_offsets,
        img_mask,
        txt_mask,
        q_weight,
        eps,
        BLOCK_T,
        BLOCK_D,
    )

    rope_dim_offsets = dim_offsets // 2
    img_rope_offsets = (
        img_token_offsets[:, None] * img_rope_stride_t + rope_dim_offsets[None, :]
    )
    txt_rope_offsets = (
        txt_token_offsets[:, None] * txt_rope_stride_t + rope_dim_offsets[None, :]
    )
    cos = _load_stream(
        img_rope_ptr,
        txt_rope_ptr,
        img_rope_offsets,
        txt_rope_offsets,
        img_mask,
        txt_mask,
    ).to(tl.float32)
    sin = _load_stream(
        img_rope_ptr,
        txt_rope_ptr,
        img_rope_offsets + BLOCK_D_HALF,
        txt_rope_offsets + BLOCK_D_HALF,
        img_mask,
        txt_mask,
    ).to(tl.float32)

    rotate_mask = (dim_offsets % 2 == 0)[None, :]
    q_rotated = _get_gptj_rotated_x(q, rotate_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF)
    q = q.to(tl.float32) * cos + q_rotated.to(tl.float32) * sin
    output_offsets = (
        token_offsets[:, None] * joint_stride_t
        + head * joint_stride_h
        + dim_offsets[None, :]
    )
    tl.store(joint_q_ptr + output_offsets, q, mask=token_mask[:, None])

    img_k_weight = tl.load(img_k_weight_ptr + weight_offsets)
    txt_k_weight = tl.load(txt_k_weight_ptr + weight_offsets)
    k_weight = tl.where(is_text[:, None], txt_k_weight, img_k_weight)
    k_offset = NUM_HEADS * BLOCK_D
    k = _load_and_norm_stream(
        img_qkv_ptr,
        txt_qkv_ptr,
        img_head_offsets + k_offset,
        txt_head_offsets + k_offset,
        img_mask,
        txt_mask,
        k_weight,
        eps,
        BLOCK_T,
        BLOCK_D,
    )
    k_rotated = _get_gptj_rotated_x(k, rotate_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF)
    k = k.to(tl.float32) * cos + k_rotated.to(tl.float32) * sin
    tl.store(joint_k_ptr + output_offsets, k, mask=token_mask[:, None])

    v_offset = 2 * NUM_HEADS * BLOCK_D
    v = _load_stream(
        img_qkv_ptr,
        txt_qkv_ptr,
        img_base_offsets + v_offset,
        txt_base_offsets + v_offset,
        img_mask,
        txt_mask,
    )
    tl.store(joint_v_ptr + output_offsets, v, mask=token_mask[:, None])

