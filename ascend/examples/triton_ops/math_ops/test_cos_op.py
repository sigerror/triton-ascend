# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import pytest
import torch
import triton
import triton.language as tl

import test_common


@triton.jit
def triton_cos(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    """

    :param in_ptr0:
    :param out_ptr0:
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
    ret = tl.cos(x0)
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    tl.store(out_ptr0 + odx, ret)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    shape_list = [
        (13, 5, 31),
    ]
    typelist = [
        "float32",
    ]
elif test_group == "full":
    shape_list = [
        (1, 1, 1),
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
def test_cos(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    shape = (L, M, N)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    if "bfloat16" == sigtype:
        x00 = x0.to(torch.float32)
        y_ref = torch.cos(x00)
        y_ref = y_ref.to(torch.bfloat16)
    elif "float16" == sigtype:
        x00 = x0.to(torch.float32)
        y_ref = torch.cos(x00)
        y_ref = y_ref.to(torch.float16)
    else:
        y_ref = torch.cos(x0)
    output = torch.zeros(shape, dtype=dtype).npu()
    triton_cos[1, 1, 1](x0, output, L, M, N)
    test_common.validate_cmp(sigtype, output, y_ref)
