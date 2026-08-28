# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math
from pathlib import Path

import pytest
import torch

import aiter

BLOCK_SIZE = 128
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def make_delta_lut(batch, heads, seqlen_q, seqlen_k, active_blocks, device):
    q_blocks = math.ceil(seqlen_q / BLOCK_SIZE)
    kv_blocks = math.ceil(seqlen_k / BLOCK_SIZE)
    assert 1 <= active_blocks < kv_blocks

    lut = torch.zeros(
        batch, heads, q_blocks, kv_blocks, dtype=torch.int32, device=device
    )
    counts = torch.full(
        (batch, heads, q_blocks),
        active_blocks,
        dtype=torch.int32,
        device=device,
    )
    selected = torch.empty(
        batch, heads, q_blocks, active_blocks, dtype=torch.int64, device=device
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(7)
    for b in range(batch):
        for h in range(heads):
            for qb in range(q_blocks):
                indices = (
                    torch.randperm(kv_blocks, generator=generator, device=device)[
                        :active_blocks
                    ]
                    .sort()
                    .values
                )
                selected[b, h, qb] = indices
                lut[b, h, qb, 0] = indices[0]
                if active_blocks > 1:
                    lut[b, h, qb, 1:active_blocks] = (indices[1:] - indices[:-1]).to(
                        torch.int32
                    )
    return lut, counts, selected


def run_vsa(q, k, v, lut, counts, kv_block_sizes, layout):
    if layout == "BHSD":
        return aiter.vsa_sparse_attention(
            q, k, v, lut, counts, kv_block_sizes
        )

    actual = aiter.vsa_sparse_attention_bshd(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        lut,
        counts,
        kv_block_sizes,
    )
    return actual.transpose(1, 2)


def masked_dense_reference(q, k, v, selected, kv_block_sizes=None):
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.size(2)
    q_blocks = math.ceil(seqlen_q / BLOCK_SIZE)

    block_mask = torch.zeros(
        batch, heads, seqlen_q, seqlen_k, dtype=torch.bool, device=q.device
    )
    for qb in range(q_blocks):
        q_start = qb * BLOCK_SIZE
        q_end = min(q_start + BLOCK_SIZE, seqlen_q)
        for b in range(batch):
            for h in range(heads):
                for block in selected[b, h, qb].tolist():
                    k_start = block * BLOCK_SIZE
                    valid_size = (
                        BLOCK_SIZE
                        if kv_block_sizes is None
                        else int(kv_block_sizes[block].item())
                    )
                    k_end = min(k_start + valid_size, seqlen_k)
                    block_mask[b, h, q_start:q_end, k_start:k_end] = True

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        q.size(-1)
    )
    probs = torch.softmax(scores.masked_fill(~block_mask, -torch.inf), dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


@pytest.mark.parametrize(
    "source_path",
    [
        "csrc/include/vsa_sparse_attention.h",
        "csrc/py_itfs_ck/vsa_sparse_attention_kernels.cu",
        "csrc/pybind/vsa_sparse_attention_pybind.cu",
    ],
)
def test_vsa_compiled_sources_are_torch_free(source_path):
    source = (REPOSITORY_ROOT / source_path).read_text()
    for forbidden in ("torch", "ATen", "TORCH_CHECK"):
        assert forbidden not in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("active_blocks", [1, 3])
@pytest.mark.parametrize("layout", ["BHSD", "BSHD"])
def test_vsa_sparse_attention_parity(dtype, active_blocks, layout):
    device = torch.device("cuda")
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 257, 481, 128
    torch.manual_seed(11)
    q = torch.randn(batch, heads, seqlen_q, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seqlen_k, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    lut, counts, selected = make_delta_lut(
        batch, heads, seqlen_q, seqlen_k, active_blocks, device
    )

    actual = run_vsa(q, k, v, lut, counts, None, layout)
    expected = masked_dense_reference(q, k, v, selected)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["BHSD", "BSHD"])
def test_vsa_sparse_attention_variable_kv_block_sizes(dtype, layout):
    device = torch.device("cuda")
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 257, 512, 128
    torch.manual_seed(13)
    q = torch.randn(batch, heads, seqlen_q, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seqlen_k, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    lut, counts, selected = make_delta_lut(
        batch, heads, seqlen_q, seqlen_k, 3, device
    )
    kv_block_sizes = torch.tensor(
        [17, 128, 73, 97], dtype=torch.int32, device=device
    )

    actual = run_vsa(q, k, v, lut, counts, kv_block_sizes, layout)
    expected = masked_dense_reference(q, k, v, selected, kv_block_sizes)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_vsa_sparse_attention_uses_current_stream():
    device = torch.device("cuda")
    q = torch.randn(1, 1, 129, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, 384, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    lut, counts, selected = make_delta_lut(1, 1, 129, 384, 2, device)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = aiter.vsa_sparse_attention(q, k, v, lut, counts)
    torch.cuda.current_stream().wait_stream(stream)

    expected = masked_dense_reference(q, k, v, selected)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda q, k, v, lut, counts: (q.cpu(), k, v, lut, counts),
            "GPU tensor",
        ),
        (
            lambda q, k, v, lut, counts: (
                q.squeeze(0),
                k,
                v,
                lut,
                counts,
            ),
            "shape \\[B, H, S, D\\]",
        ),
        (
            lambda q, k, v, lut, counts: (q.float(), k, v, lut, counts),
            "float16 or bfloat16",
        ),
        (
            lambda q, k, v, lut, counts: (
                q.transpose(2, 3),
                k,
                v,
                lut,
                counts,
            ),
            "contiguous BHSD",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k.to(torch.bfloat16),
                v,
                lut,
                counts,
            ),
            "same dtype",
        ),
        (
            lambda q, k, v, lut, counts: (
                q[:0],
                k[:0],
                v[:0],
                lut,
                counts,
            ),
            "positive",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k[:, :0],
                v[:, :0],
                lut,
                counts,
            ),
            "positive",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k.expand(2, -1, -1, -1).contiguous(),
                v.expand(2, -1, -1, -1).contiguous(),
                lut,
                counts,
            ),
            "same batch size",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k,
                v[:, :, :-1],
                lut,
                counts,
            ),
            "same shape",
        ),
        (
            lambda q, k, v, lut, counts: (
                q.expand(-1, 3, -1, -1).contiguous(),
                k.expand(-1, 2, -1, -1).contiguous(),
                v.expand(-1, 2, -1, -1).contiguous(),
                lut,
                counts,
            ),
            "divisible",
        ),
        (
            lambda q, k, v, lut, counts: (
                q[..., :64].contiguous(),
                k,
                v,
                lut,
                counts,
            ),
            "head dimension 128",
        ),
        (
            lambda q, k, v, lut, counts: (
                q[:, :, :0],
                k,
                v,
                lut,
                counts,
            ),
            "query length must be positive",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k[:, :, :128],
                v[:, :, :128],
                lut,
                counts,
            ),
            "key length must exceed 128",
        ),
        (
            lambda q, k, v, lut, counts: (q, k, v, lut.cpu(), counts),
            "GPU tensor",
        ),
        (
            lambda q, k, v, lut, counts: (q, k, v, lut, counts.cpu()),
            "GPU tensor",
        ),
        (
            lambda q, k, v, lut, counts: (q, k, v, lut.long(), counts),
            "dtype int32",
        ),
        (
            lambda q, k, v, lut, counts: (q, k, v, lut, counts.long()),
            "dtype int32",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k,
                v,
                lut.expand(2, -1, -1, -1),
                counts,
            ),
            "contiguous",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k,
                v,
                lut,
                counts.expand(2, -1, -1),
            ),
            "contiguous",
        ),
        (
            lambda q, k, v, lut, counts: (q, k, v, lut[..., :-1], counts),
            "block_lut must have shape",
        ),
        (
            lambda q, k, v, lut, counts: (q, k, v, lut, counts[..., :0]),
            "block_counts must have shape",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k,
                v,
                lut,
                counts.zero_(),
            ),
            "at least one KV block",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k,
                v,
                lut,
                counts.fill_(lut.size(-1)),
            ),
            "final slot is reserved",
        ),
    ],
)
def test_vsa_sparse_attention_validation(mutate, match):
    device = torch.device("cuda")
    q = torch.randn(1, 1, 128, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, 384, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    lut, counts, _ = make_delta_lut(1, 1, 128, 384, 1, device)

    args = mutate(q, k, v, lut, counts)
    with pytest.raises(RuntimeError, match=match):
        aiter.vsa_sparse_attention(*args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("caller_out", [False, True])
def test_vsa_sparse_attention_bshd_accepts_mixed_physical_layout(
    dtype, caller_out
):
    device = torch.device("cuda")
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 257, 481, 128
    torch.manual_seed(17)
    q = torch.randn(batch, heads, seqlen_q, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seqlen_k, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    lut, counts, selected = make_delta_lut(
        batch, heads, seqlen_q, seqlen_k, 3, device
    )

    q_bshd = q.transpose(1, 2)
    k_bshd = k.transpose(1, 2)
    v_bshd = v.transpose(1, 2).contiguous()
    out = torch.empty_like(q_bshd, memory_format=torch.contiguous_format)
    actual_bshd = aiter.vsa_sparse_attention_bshd(
        q_bshd,
        k_bshd,
        v_bshd,
        lut,
        counts,
        out=out if caller_out else None,
    )

    assert not q_bshd.is_contiguous()
    assert not k_bshd.is_contiguous()
    assert v_bshd.is_contiguous()
    assert actual_bshd.is_contiguous()
    if caller_out:
        assert actual_bshd.data_ptr() == out.data_ptr()

    expected = masked_dense_reference(q, k, v, selected)
    torch.testing.assert_close(
        actual_bshd.transpose(1, 2), expected, atol=3e-2, rtol=3e-2
    )


def test_vsa_sparse_attention_bshd_rejects_strided_last_dimension():
    device = torch.device("cuda")
    q = torch.randn(1, 128, 2, 256, device=device, dtype=torch.float16)[
        ..., ::2
    ]
    k = torch.randn(1, 384, 2, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    lut, counts, _ = make_delta_lut(1, 2, 128, 384, 1, device)

    with pytest.raises(RuntimeError, match="contiguous last dimension"):
        aiter.vsa_sparse_attention_bshd(q, k, v, lut, counts)


def test_vsa_sparse_attention_bshd_validates_caller_output():
    device = torch.device("cuda")
    q = torch.randn(1, 128, 2, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 384, 2, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    out = torch.empty(1, 128, 2, 256, device=device, dtype=q.dtype)[..., ::2]
    lut, counts, _ = make_delta_lut(1, 2, 128, 384, 1, device)

    with pytest.raises(RuntimeError, match="contiguous last dimension"):
        aiter.vsa_sparse_attention_bshd(q, k, v, lut, counts, out=out)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda sizes: sizes.cpu(), "GPU tensor"),
        (lambda sizes: sizes.long(), "dtype int32"),
        (lambda sizes: sizes.expand(2, -1), "contiguous"),
        (lambda sizes: sizes[:-1], "shape \\[ceil\\(Sk/128\\)\\]"),
        (lambda sizes: sizes.clone().fill_(0), "between 1 and 128"),
        (lambda sizes: sizes.clone().fill_(129), "between 1 and 128"),
    ],
)
def test_vsa_sparse_attention_validates_kv_block_sizes(mutate, match):
    device = torch.device("cuda")
    q = torch.randn(1, 1, 128, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, 384, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    lut, counts, _ = make_delta_lut(1, 1, 128, 384, 1, device)
    kv_block_sizes = torch.full(
        (3,), BLOCK_SIZE, dtype=torch.int32, device=device
    )

    with pytest.raises(RuntimeError, match=match):
        aiter.vsa_sparse_attention(
            q, k, v, lut, counts, mutate(kv_block_sizes)
        )


def test_vsa_sparse_attention_rejects_cross_device_kv_block_sizes():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    q = torch.randn(1, 1, 128, 128, device="cuda:0", dtype=torch.float16)
    k = torch.randn(1, 1, 384, 128, device="cuda:0", dtype=torch.float16)
    v = torch.randn_like(k)
    lut, counts, _ = make_delta_lut(1, 1, 128, 384, 1, q.device)
    kv_block_sizes = torch.full(
        (3,), BLOCK_SIZE, dtype=torch.int32, device="cuda:1"
    )

    with pytest.raises(RuntimeError, match="same GPU"):
        aiter.vsa_sparse_attention(q, k, v, lut, counts, kv_block_sizes)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("tensor_name", ["k", "block_lut"])
def test_vsa_sparse_attention_rejects_cross_device_inputs(tensor_name):
    q = torch.randn(1, 1, 128, 128, device="cuda:0", dtype=torch.float16)
    k = torch.randn(1, 1, 384, 128, device="cuda:0", dtype=torch.float16)
    v = torch.randn_like(k)
    lut, counts, _ = make_delta_lut(1, 1, 128, 384, 1, q.device)

    if tensor_name == "k":
        k = k.to("cuda:1")
        v = v.to("cuda:1")
    else:
        lut = lut.to("cuda:1")

    with pytest.raises(RuntimeError, match="same GPU"):
        aiter.vsa_sparse_attention(q, k, v, lut, counts)
