# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import math
import triton
import triton.language as tl
import test_common
import torch
import pytest


def expand_to_next_power_of_two(a):
    """
    _summary_

    :param a:
    :raises ValueError:
    :return:
    """
    if a <= 0:
        raise ValueError("must >0")
    if (math.log2(a)).is_integer():
        return a
    return 2**math.ceil(math.log2(a))


@triton.jit
def fn_npu(output_ptr, x_ptr, X: tl.constexpr, Y: tl.constexpr,
           Z: tl.constexpr, XNUMEL: tl.constexpr, YNUMEL: tl.constexpr,
           ZNUMEL: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param x_ptr:
    :param X:
    :param Y:
    :param Z:
    :param XNUMEL:
    :param YNUMEL:
    :param ZNUMEL:
    """
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :,
                                              None] * Z + zidx[None, None, :]
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None,
                                                                    None, :])
    abc = tl.load(x_ptr + oidx, mask=mask)
    ret = tl.zeros_like(abc)
    tl.store(output_ptr + oidx, ret, mask=mask)


testlist = [
    (fn_npu, 'int8', 2, 255, 9),
    (fn_npu, 'int16', 3, 5, 3),
    (fn_npu, 'int32', 2, 255, 9),
    (fn_npu, 'float16', 55, 5, 16),
    (fn_npu, 'float16', 4, 5, 17),
    (fn_npu, 'float16', 6, 5, 15),
    (fn_npu, 'float16', 2, 1928, 3),
    (fn_npu, 'float32', 2, 255, 9),
    (fn_npu, 'bool', 3, 5, 3),
]


@pytest.mark.parametrize('testfunc, sigtype, X, Y, Z', testlist)
def test_zeros_like(testfunc, sigtype, X, Y, Z):
    """
    _summary_

    :param testfunc:
    :param sigtype:
    :param X:
    :param Y:
    :param Z:
    """
    dtype = test_common.get_torch_typename(sigtype)
    XNUMEL = expand_to_next_power_of_two(X)
    YNUMEL = expand_to_next_power_of_two(Y)
    ZNUMEL = expand_to_next_power_of_two(Z)
    x = torch.full((X, Y, Z), 10, dtype=dtype).npu()
    y = torch.full((X, Y, Z), 0, dtype=dtype).npu()
    output = torch.full((X, Y, Z), 5, dtype=dtype).npu()
    testfunc[1, 1, 1](output, x, X, Y, Z, XNUMEL, YNUMEL, ZNUMEL)
    test_common.validate_cmp(sigtype, output, y)
