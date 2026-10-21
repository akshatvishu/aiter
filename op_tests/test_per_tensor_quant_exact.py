# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter import dtypes
from aiter.ops.quant import dynamic_per_tensor_quant, static_per_tensor_quant


@pytest.mark.parametrize(
    "shape", [(21, 3072), (512, 3072), (513, 3072), (4096, 3072), (512, 3088)]
)
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("output_dtype", [dtypes.fp8, torch.int8])
@pytest.mark.parametrize("zero", [False, True])
def test_dynamic_per_tensor_exact(shape, input_dtype, output_dtype, zero):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=input_dtype)
    if zero:
        x.zero_()
    else:
        # Put the maximum at the end, beyond the first reduction chunk.
        x[-1, -1] = -20.0
    limit = (
        torch.finfo(output_dtype).max
        if output_dtype.is_floating_point
        else torch.iinfo(output_dtype).max
    )
    inverse_limit = torch.tensor(1.0 / limit, dtype=torch.float32, device="cuda")
    reference_scale = (x.abs().amax().float() * inverse_limit).reshape(1)
    reference = torch.empty_like(x, dtype=output_dtype)
    static_per_tensor_quant(reference, x, reference_scale)
    # An offset output exercises the fallback when scratch would be unaligned.
    for offset in (0, 1):
        storage = torch.full((x.numel() + 32,), 16, dtype=output_dtype, device="cuda")
        out = storage[offset : offset + x.numel()].view(shape)
        scale = torch.empty_like(reference_scale)
        for _ in range(3):
            out.fill_(16)
            scale.fill_(99.0)
            dynamic_per_tensor_quant(out, x, scale)
            assert torch.equal(
                scale.view(torch.int32), reference_scale.view(torch.int32)
            )
            assert torch.equal(out.view(torch.uint8), reference.view(torch.uint8))
        assert bool((storage[:offset].float() == 16).all())
        assert bool((storage[offset + x.numel() :].float() == 16).all())
