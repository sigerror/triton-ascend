# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pytest
import test_common
import torch
import triton
import triton.language as tl


@triton.jit
def triton_div(
    in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr
):
    """

    :param in_ptr0:
    :param in_ptr1:
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
    x1 = tl.load(in_ptr1 + idx)
    ret = x0 / x1
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    tl.store(out_ptr0 + odx, ret)


# always use full test_group
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
    "bool",
]


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
def test_div(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    shape = (L, M, N)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype)
    x1 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype)
    x1 = test_common.fill_zero_with_one(x1)

    y_ref = torch.div(x0, x1)
    if "int8" == sigtype:
        y_ref = y_ref.to(torch.int8)
    elif "int16" == sigtype:
        y_ref = y_ref.to(torch.int16)
    elif "int32" == sigtype:
        y_ref = y_ref.to(torch.int32)
    elif "int64" == sigtype:
        y_ref = y_ref.to(torch.int64)
    dtype = test_common.get_torch_typename(sigtype)
    output = torch.zeros(shape, dtype=dtype).npu()
    y_ref = y_ref.npu()
    triton_div[1, 1, 1](x0.npu(), x1.npu(), output, L, M, N)

    test_common.validate_cmp(sigtype, output, y_ref)
