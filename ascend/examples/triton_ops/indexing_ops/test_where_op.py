# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import pytest
import torch
import triton
import triton.language as tl

import test_common


def torch_where(x0, x1):
    """

    :param x0:
    :param x1:

    """
    res = torch.where(x0 < x1, x0, x1)
    return res


@triton.jit
def triton_where(
    output_ptr, x_ptr, y_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr
):
    """

    :param output_ptr:
    :param x_ptr:
    :param y_ptr:
    :param XB: tl.constexpr:
    :param YB: tl.constexpr:
    :param ZB: tl.constexpr:

    """
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)
    tmp2 = X < Y
    ret = tl.where(tmp2, X, Y)
    tl.store(output_ptr + idx, ret)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    shape_list = [
        (37, 5, 3),
    ]
    typelist = [
        "float32",
    ]
elif test_group == "full":
    shape_list = [
        (1, 1, 1),
        (1, 22, 39),
        (27, 1, 39),
        (27, 22, 1),
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
        (7, 11, 32),
        (7, 11, 33),
        (2, 3, 255),
        (3, 3, 256),
        (3, 2, 257),
    ]
    typelist = [
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "bool",
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
                sigtype in ["int64", "bfloat16"], reason="Unsupported for now"
            ),
        )
        for sigtype in typelist
    ],
)
def test_where(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    ans = torch_where(x0, x1)
    output = torch.zeros((L, M, N), dtype=dtype).npu()
    triton_where[1, 1, 1](output, x0, x1, L, M, N)
    test_common.validate_cmp(sigtype, output, ans)
