# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import pytest
import torch
import triton
import triton.language as tl

import test_common


@triton.jit
def triton_softmax(
    in_ptr0, out_ptr0, axis: tl.constexpr, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr
):
    """

    :param in_ptr0:
    :param out_ptr0:
    :param axis: tl.constexpr:
    :param L: tl.constexpr:
    :param M: tl.constexpr:
    :param N: tl.constexpr:

    """
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    x0 = tl.load(in_ptr0 + idx)
    ret = tl.softmax(x0, dim=axis, keep_dims=True)
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    tl.store(out_ptr0 + odx, ret)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    axis_list = [0, 1, 2]
    shape_list = [
        (1, 1, 1),
        (13, 5, 31),
    ]
    typelist = [
        "float32",
    ]
elif test_group == "full":
    axis_list = [0, 1, 2]
    shape_list = [
        (1, 1, 1),
        (1, 1, 23),
        (23, 1, 1),
        (1, 23, 1),
        (37, 5, 3),
        (2, 29, 4),
        (7, 31, 7),
        (3, 5, 8),
        (7, 17, 15),
        (25, 5, 16),
        (23, 5, 31),
        (19, 11, 32),
        (7, 11, 33),
        (2, 3, 255),
        (3, 3, 256),
        (3, 2, 257),
    ]
    typelist = [
        "float16",
        "bfloat16",
        "float32",
    ]
else:
    raise ValueError(f"Invalid TRITON_TEST_GROUP {test_group}")


@pytest.mark.parametrize("axis", axis_list)
@pytest.mark.parametrize("L, M, N", shape_list)
@pytest.mark.parametrize(
    "sigtype",
    [
        pytest.param(
            sigtype,
            marks=pytest.mark.skipif(
                sigtype in ["bfloat16"], reason="Unsupported for now"
            ),
        )
        for sigtype in typelist
    ],
)
def test_softmax(sigtype, axis, L, M, N):
    """

    :param sigtype:
    :param axis:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    shape = (L, M, N)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    softmax_layer = torch.nn.Softmax(dim=axis)
    y_ref = softmax_layer(x0)
    output = torch.zeros(shape, dtype=dtype).npu()
    triton_softmax[1, 1, 1](x0, output, axis, L, M, N)
    test_common.validate_cmp(sigtype, output, y_ref)
