# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import pytest
import torch
import triton
import triton.language as tl

import test_common


@triton.jit
def triton_max_dim2(
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
    ret = tl.max(x, -1)
    odx = lblk_idx[:, None] * M + mblk_idx[None, :]
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
        (5, 8, 7),
        (27, 8, 2),
        (3, 2, 5),
        (6, 9, 2),
        (1, 22, 39),
        (27, 1, 39),
        (27, 22, 1),
        (1, 1, 39),
        (1, 22, 1),
        (27, 1, 1),
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
def test_max_2(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    if "int" in sigtype:
        ans = torch.max(x0.to(torch.int64), 2)[0].to(dtype)
    else:
        ans = torch.max(x0, 2)[0]
    output = torch.zeros((L, M), dtype=dtype).npu()
    triton_max_dim2[1, 1, 1](x0, output, L, M, N)
    test_common.validate_cmp(sigtype, output, ans)
