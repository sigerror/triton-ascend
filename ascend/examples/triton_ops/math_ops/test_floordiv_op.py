# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import pytest
import test_common
import torch
import triton
import triton.language as tl


@triton.jit
def triton_floordiv_i8(
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
    x0 = tl.load(in_ptr0 + idx).to(tl.float32)
    x1 = tl.load(in_ptr1 + idx).to(tl.float32)
    ret = tl.floor(x0 / x1)
    ret.to(tl.int8)
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_floordiv_i16(
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
    x0 = tl.load(in_ptr0 + idx).to(tl.float32)
    x1 = tl.load(in_ptr1 + idx).to(tl.float32)
    ret = tl.floor(x0 / x1)
    ret1 = ret.to(tl.int16)
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    tl.store(out_ptr0 + odx, ret1)


@triton.jit
def triton_floordiv_i32(
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
    x0 = tl.load(in_ptr0 + idx).to(tl.float32)
    x1 = tl.load(in_ptr1 + idx).to(tl.float32)
    ret = tl.floor(x0 / x1)
    ret1 = ret.to(tl.int32)
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    tl.store(out_ptr0 + odx, ret1)


@triton.jit
def triton_floordiv_i64(
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
    x0 = tl.load(in_ptr0 + idx).to(tl.float32)
    x1 = tl.load(in_ptr1 + idx).to(tl.float32)
    ret = tl.floor(x0 / x1)
    ret1 = ret.to(tl.int64)
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    tl.store(out_ptr0 + odx, ret1)


@triton.jit
def triton_floordiv_float(
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
    ret1 = tl.floor(x0 / x1)
    odx = (
        lblk_idx[:, None, None] * N * M
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
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
def test_floordiv(sigtype, L, M, N):
    """

    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    dtype = test_common.get_torch_typename(sigtype)
    shape = (L, M, N)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype)
    x1 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype)
    x1 = test_common.fill_zero_with_one(x1)
    output = torch.zeros(shape, dtype=dtype).npu()

    if "int8" in sigtype:
        x1[0][0][0] = 0
        y_ref = torch.floor(x0 / x1).to(torch.int8).npu()
        triton_floordiv_i8[1, 1, 1](x0.npu(), x1.npu(), output, L, M, N)
    elif "int16" in sigtype:
        x1[0][0][0] = 0
        y_ref = torch.floor(x0 / x1).to(torch.int16).npu()
        triton_floordiv_i16[1, 1, 1](x0.npu(), x1.npu(), output, L, M, N)
    elif "int32" in sigtype:
        x1[0][0][0] = 0
        y_ref = torch.floor(x0 / x1).to(torch.int32).npu()
        triton_floordiv_i32[1, 1, 1](x0.npu(), x1.npu(), output, L, M, N)
    elif "int64" in sigtype:
        x1[0][0][0] = 0
        y_ref = torch.floor(x0 / x1).to(torch.int64).npu()
        triton_floordiv_i64[1, 1, 1](x0.npu(), x1.npu(), output, L, M, N)
    else:
        y_ref = x0 / x1
        triton_floordiv_float[1, 1, 1](x0, x1, output, L, M, N)
    test_common.validate_cmp(sigtype, output, y_ref)
