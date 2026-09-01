# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

import aiter


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_interleaved: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    if is_interleaved:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(
            -2
        )

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    is_interleaved: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos_sin.chunk(2, dim=-1)
    q = F.rms_norm(q, (q.shape[-1],), q_weight, eps)
    k = F.rms_norm(k, (k.shape[-1],), k_weight, eps)
    return _apply_rope(q, cos, sin, is_interleaved), _apply_rope(
        k, cos, sin, is_interleaved
    )


@pytest.mark.parametrize(
    (
        "batch_size",
        "is_interleaved",
        "packed",
        "tokens0",
        "tokens1",
        "num_heads_q",
        "num_heads_k",
    ),
    [
        (1, True, True, 5, 1024, 24, 24),
        (1, True, True, 13, 1024, 24, 24),
        (2, True, True, 13, 1024, 24, 24),
        (1, False, True, 13, 1024, 24, 24),
        (1, True, False, 13, 1024, 24, 24),
        (1, False, False, 13, 1024, 24, 24),
        (1, True, False, 13, 1024, 24, 8),
    ],
)
def test_fused_qk_norm_rope_2way_matches_reference(
    batch_size: int,
    is_interleaved: bool,
    packed: bool,
    tokens0: int,
    tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
) -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    head_size = 128
    q_hidden_size = num_heads_q * head_size
    k_hidden_size = num_heads_k * head_size
    eps = 1e-6

    def make_qkv(tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not packed:
            return (
                torch.randn(
                    batch_size,
                    tokens,
                    num_heads_q,
                    head_size,
                    device=device,
                    dtype=dtype,
                ),
                torch.randn(
                    batch_size,
                    tokens,
                    num_heads_k,
                    head_size,
                    device=device,
                    dtype=dtype,
                ),
                torch.randn(
                    batch_size,
                    tokens,
                    num_heads_k,
                    head_size,
                    device=device,
                    dtype=dtype,
                ),
            )
        assert num_heads_q == num_heads_k
        qkv = torch.randn(
            batch_size,
            tokens,
            q_hidden_size + 2 * k_hidden_size,
            device=device,
            dtype=dtype,
        )
        q, k, v = qkv.split((q_hidden_size, k_hidden_size, k_hidden_size), dim=-1)
        return (
            q.unflatten(-1, (num_heads_q, head_size)),
            k.unflatten(-1, (num_heads_k, head_size)),
            v.unflatten(-1, (num_heads_k, head_size)),
        )

    q0, k0, v0 = make_qkv(tokens0)
    q1, k1, v1 = make_qkv(tokens1)
    q0_before, k0_before, v0_before = q0.clone(), k0.clone(), v0.clone()
    q1_before, k1_before, v1_before = q1.clone(), k1.clone(), v1.clone()

    q_weight0 = torch.randn(head_size, device=device, dtype=dtype)
    k_weight0 = torch.randn(head_size, device=device, dtype=dtype)
    q_weight1 = torch.randn(head_size, device=device, dtype=dtype)
    k_weight1 = torch.randn(head_size, device=device, dtype=dtype)
    cos_sin0 = torch.randn(tokens0, head_size, device=device, dtype=dtype)
    cos_sin1 = torch.randn(tokens1, head_size, device=device, dtype=dtype)

    q0_ref, k0_ref = _reference(
        q0, k0, q_weight0, k_weight0, cos_sin0, is_interleaved, eps
    )
    q1_ref, k1_ref = _reference(
        q1, k1, q_weight1, k_weight1, cos_sin1, is_interleaved, eps
    )
    q_ref = torch.cat((q0_ref, q1_ref), dim=1)
    k_ref = torch.cat((k0_ref, k1_ref), dim=1)
    q_out = torch.empty_like(q_ref)
    k_out = torch.empty_like(k_ref)

    aiter.fused_qk_norm_rope_2way(
        q0,
        k0,
        q1,
        k1,
        q_weight0,
        k_weight0,
        q_weight1,
        k_weight1,
        cos_sin0,
        cos_sin1,
        batch_size,
        tokens0,
        tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        q_out,
        k_out,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=0.05)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=0.05)
    assert torch.equal(q0, q0_before)
    assert torch.equal(k0, k0_before)
    assert torch.equal(v0, v0_before)
    assert torch.equal(q1, q1_before)
    assert torch.equal(k1, k1_before)
    assert torch.equal(v1, v1_before)


@pytest.mark.parametrize("invalid_stride", ["head", "last_dim"])
def test_fused_qk_norm_rope_2way_rejects_noncontiguous_heads(
    invalid_stride: str,
) -> None:
    batch_size = 1
    tokens0 = 5
    tokens1 = 7
    num_heads = 2
    head_size = 128
    dtype = torch.bfloat16
    device = "cuda"

    if invalid_stride == "head":
        q0 = torch.randn(
            batch_size,
            tokens0,
            num_heads,
            head_size + 1,
            dtype=dtype,
            device=device,
        )[..., :head_size]
    else:
        q0 = torch.randn(
            batch_size,
            tokens0,
            num_heads,
            2 * head_size,
            dtype=dtype,
            device=device,
        )[..., ::2]

    k0 = torch.randn_like(q0).contiguous()
    q1 = torch.randn(
        batch_size,
        tokens1,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    k1 = torch.randn_like(q1)
    weights = [torch.randn(head_size, dtype=dtype, device=device) for _ in range(4)]
    cos_sin0 = torch.randn(tokens0, head_size, dtype=dtype, device=device)
    cos_sin1 = torch.randn(tokens1, head_size, dtype=dtype, device=device)
    q_out = torch.empty(
        batch_size,
        tokens0 + tokens1,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    k_out = torch.empty_like(q_out)

    with pytest.raises(RuntimeError, match="contiguous within each head"):
        aiter.fused_qk_norm_rope_2way(
            q0,
            k0,
            q1,
            k1,
            *weights,
            cos_sin0,
            cos_sin1,
            batch_size,
            tokens0,
            tokens1,
            num_heads,
            num_heads,
            head_size,
            True,
            1e-6,
            q_out,
            k_out,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
