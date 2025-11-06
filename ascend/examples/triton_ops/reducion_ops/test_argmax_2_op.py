# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pytest
import torch
import triton
import triton.language as tl

import test_common


@triton.jit
def triton_argmax_dim2(
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
    ret = tl.argmax(x, 2)
    odx = lblk_idx[:, None] * M + mblk_idx[None, :]
    tl.store(out_ptr0 + odx, ret)


test_group = "full"
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
def test_argmax_2(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    torch.manual_seed(0)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    if "int" in sigtype:
        ans = torch.max(x0.to(torch.int64), 2)[1]
    else:
        ans = torch.max(x0, 2)[1]
    output = torch.zeros((L, M), dtype=torch.int32).npu()
    triton_argmax_dim2[1, 1, 1](x0, output, L, M, N)
    test_common.validate_cmp("int32", output, ans)
