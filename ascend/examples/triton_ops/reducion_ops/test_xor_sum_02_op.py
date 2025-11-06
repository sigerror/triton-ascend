# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import pytest
import torch
import triton
import triton.language as tl

import test_common


def torch_func(tensor, dim):
    """

    :param tensor:
    :param dim:

    """
    dim = dim if dim >= 0 else tensor.dim() + dim
    num_slices = tensor.size(dim)
    result = tensor.select(dim, 0)
    for i in range(1, num_slices):
        result = torch.bitwise_xor(result, tensor.select(dim, i))

    return result


@triton.jit
def triton_xor_sum_dim0_dim2(
    in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr
):
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
    x = tl.load(in_ptr0 + idx)
    ret = tl.xor_sum(x, 0)
    ret1 = tl.xor_sum(ret, 1)
    odx = mblk_idx[:]
    tl.store(out_ptr0 + odx, ret1)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    shape_list = [
        (13, 5, 31),
    ]
    typelist = [
        "int32",
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
                sigtype in ["int64"], reason="Unsupported for now"
            ),
        )
        for sigtype in typelist
    ],
)
def test_xor_sum_02(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    ans = torch_func(x0, 0)
    ans = torch_func(ans, 1)
    output = torch.zeros((M,), dtype=dtype).npu()
    triton_xor_sum_dim0_dim2[1, 1, 1](x0, output, L, M, N)
    test_common.validate_cmp(sigtype, output, ans)
