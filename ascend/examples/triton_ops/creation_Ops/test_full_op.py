# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import triton
import triton.language as tl
import test_common
import torch
import pytest


@triton.jit
def fn_npu_int8(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param X:
    :param Y:
    :param Z:
    """
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.int8)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :,
                                              None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int16(output_ptr, X: tl.constexpr, Y: tl.constexpr,
                 Z: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param X:
    :param Y:
    :param Z:
    """
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.int16)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :,
                                              None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int32(output_ptr, X: tl.constexpr, Y: tl.constexpr,
                 Z: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param X:
    :param Y:
    :param Z:
    """
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.int32)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :,
                                              None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp16(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param X:
    :param Y:
    :param Z:
    """
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.float16)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :,
                                              None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp32(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param X:
    :param Y:
    :param Z:
    """
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.float32)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :,
                                              None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_bool(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param X:
    :param Y:
    :param Z:
    """
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.int1)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :,
                                              None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


testlist = [
    (fn_npu_int8, 'int8', 2, 255, 9),
    (fn_npu_int16, 'int16', 3, 5, 3),
    (fn_npu_int32, 'int32', 2, 255, 9),
    (fn_npu_fp16, 'float16', 55, 5, 16),
    (fn_npu_fp16, 'float16', 4, 5, 17),
    (fn_npu_fp16, 'float16', 6, 5, 15),
    (fn_npu_fp16, 'float16', 2, 1928, 3),
    (fn_npu_fp32, 'float32', 2, 255, 9),
    (fn_npu_bool, 'bool', 3, 5, 3),
]


@pytest.mark.parametrize('testfunc, sigtype, X, Y, Z', testlist)
def test_full(testfunc, sigtype, X, Y, Z):
    """
    _summary_

    :param testfunc:
    :param sigtype:
    :param X:
    :param Y:
    :param Z:
    """
    dtype = test_common.get_torch_typename(sigtype)
    x = torch.full((X, Y, Z), 100, dtype=dtype).npu()
    output = torch.randint(1, (X, Y, Z), dtype=dtype).npu()
    testfunc[1, 1, 1](output, X, Y, Z)
    test_common.validate_cmp(sigtype, output, x)
