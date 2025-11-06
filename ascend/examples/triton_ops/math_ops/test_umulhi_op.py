# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import test_common


@triton.jit
def triton_umulhi(
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
    ret = tl.umulhi(x0, x1)
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    tl.store(out_ptr0 + odx, ret)


def umulhi_dtype(a, b, dtype):
    """

    :param a:
    :param b:
    :param dtype:

    """
    if "int8" in dtype:
        a_16 = a.astype(np.int16)
        b_16 = b.astype(np.int16)
        product_16 = a_16 * b_16
        result_high_8 = product_16 >> 8
        result_high = result_high_8.astype(np.int8)
    if "int16" in dtype:
        a_32 = a.astype(np.int32)
        b_32 = b.astype(np.int32)
        product_32 = a_32 * b_32
        result_high_16 = product_32 >> 16
        result_high = result_high_16.astype(np.int16)
    if "int32" in dtype:
        a_64 = a.astype(np.int64)
        b_64 = b.astype(np.int64)
        product_64 = a_64 * b_64
        result_high_32 = product_64 >> 32
        result_high = result_high_32.astype(np.int32)
    return result_high


testlist = [
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
    "int32",
]


@pytest.mark.parametrize("L, M, N", testlist)
@pytest.mark.parametrize("sigtype", typelist)
def test_umulhi(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    max_val = 2000
    x = torch.randint(low=0, high=max_val, size=(L, M, N), dtype=dtype)
    y = torch.randint(low=0, high=max_val, size=(L, M, N), dtype=dtype)
    xx = x.npu()
    yy = y.npu()
    z_tri = torch.zeros(size=(L, M, N), dtype=dtype).npu()
    triton_umulhi[1, 1, 1](xx, yy, z_tri, L, M, N)
    xxx = x.numpy()
    yyy = y.numpy()
    z_ref = umulhi_dtype(xxx, yyy, sigtype)
    z_ref1 = torch.from_numpy(z_ref).npu()
    torch.equal(z_tri, z_ref1)
