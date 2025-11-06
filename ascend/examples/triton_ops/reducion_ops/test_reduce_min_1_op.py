# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import pytest
import torch
import triton
import triton.language as tl

import test_common


@triton.jit
def promote_to_tensor(x):
    """

    :param x:

    """
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def is_floating(x):
    """

    :param x:

    """
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    """

    :param a_value:
    :param a_index:
    :param b_value:
    :param b_index:

    """
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def triton_min_dim1(
    in_ptr0, in_ptr1, out_ptr0, out_ptr1, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr
):
    """

    :param in_ptr0:
    :param in_ptr1:
    :param out_ptr0:
    :param out_ptr1:
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
    odx = lblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret, ret1 = tl.reduce((x, x1), 1, minimum_with_index)
    tl.store(out_ptr0 + odx, ret)
    tl.store(out_ptr1 + odx, ret1)


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
        (13, 5, 31),
        (9, 11, 32),
        (7, 11, 33),
    ]
    typelist = [
        "int8",
        "int16",
        "int32",
        "int64",
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
                sigtype in ["int64", "bfloat16"], reason="Unsupported for now"
            ),
        )
        for sigtype in typelist
    ],
)
def test_reduce_min_1(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(L, M, N), dtype="int32").npu()
    if "int" in sigtype:
        ans, ans1 = torch.min(x0.to(torch.int64), 1)
        ans = ans.to(dtype)
    else:
        ans, ans1 = torch.min(x0, 1)
    output = torch.zeros((L, N), dtype=dtype).npu()
    output1 = torch.zeros((L, N), dtype=torch.int32).npu()
    triton_min_dim1[1, 1, 1](x0, x1, output, output1, L, M, N)
    test_common.validate_cmp(sigtype, output, ans)
    test_common.validate_cmp("int32", output1, ans1)
