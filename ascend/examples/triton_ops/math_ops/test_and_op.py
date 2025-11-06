# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
import pytest
import test_common
import torch
import triton
import triton.language as tl


@triton.jit
def triton_and(
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
    ret = x0 & x1
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
    (23, 5, 31),
    (19, 11, 32),
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
def test_and(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    shape = (L, M, N)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    if "float" in sigtype:
        x0 = x0.to(torch.int32)
        x1 = x1.to(torch.int32)
        y_ref = x0 & x1
        y_ref = y_ref.to(dtype)
    else:
        y_ref = x0 & x1
    output = torch.zeros(shape, dtype=dtype).npu()
    triton_and[1, 1, 1](x0, x1, output, L, M, N)
    test_common.validate_cmp(sigtype, output, y_ref)
