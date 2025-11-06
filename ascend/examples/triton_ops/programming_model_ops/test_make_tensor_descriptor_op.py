# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    In,
    Out,
    in_shape1: tl.constexpr,
    in_shape2: tl.constexpr,
    in_shape3: tl.constexpr,
    ou_shape1: tl.constexpr,
    ou_shape2: tl.constexpr,
    axis: tl.constexpr,
):
    """_summary_

    :param In: 
    :param Out: 
    :param in_shape1: 
    :param in_shape2: 
    :param in_shape3: 
    :param ou_shape1: 
    :param ou_shape2: 
    :param axis: 
    """
    in_desc = tl.make_tensor_descriptor(
        base=In,
        shape=[in_shape1 * in_shape2 * in_shape3],
        strides=[1],
        block_shape=[in_shape1 * in_shape2 * in_shape3],
    )
    out_desc = tl.make_tensor_descriptor(
        base=Out,
        shape=[ou_shape1 * ou_shape2],
        strides=[1],
        block_shape=[ou_shape1 * ou_shape2],
    )
    val = in_desc.load([0]).reshape(in_shape1, in_shape2, in_shape3)
    output = tl.max(val, axis=axis)
    out_desc.store([0], output.reshape(out_desc.block_shape))


@pytest.mark.parametrize("dtype_str", ["int32"])
@pytest.mark.parametrize("shapelist", [(128, 2, 4), (64, 2, 4), (32, 2, 4),
                                       (2, 4, 32), (2, 4, 2)])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("device", ["npu"])
def test_make_tensor_descriptor(dtype_str, shapelist, axis, device):
    """_summary_

    :param dtype_str: 
    :param shape: 
    :param axis: 
    :param device: 
    """
    inp = torch.arange(math.prod(shapelist),
                       dtype=getattr(torch, dtype_str),
                       device=device).reshape(shapelist)
    expected, indices = torch.max(inp.to(torch.int64), dim=axis)
    indices = indices
    expected = expected.to(torch.int32)
    actual = torch.zeros(expected.shape,
                         dtype=getattr(torch, dtype_str),
                         device=device)
    kernel[(1, )](inp, actual, *shapelist, *expected.shape, axis=axis)
    assert torch.equal(expected, actual)
