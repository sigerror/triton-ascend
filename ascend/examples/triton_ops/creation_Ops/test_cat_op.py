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
    """
    if a <= 0:
        raise ValueError("must >0")
    if (math.log2(a)).is_integer():
        return a
    return 2**math.ceil(math.log2(a))


@triton.jit
def fn_npu(output_ptr, x_ptr, y_ptr, X: tl.constexpr, Y: tl.constexpr,
           Z: tl.constexpr, XNUMEL: tl.constexpr, YNUMEL: tl.constexpr,
           ZNUMEL: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param x_ptr:
    :param y_ptr:
    :param X:
    :param Y:
    :param Z:
    :param XNUMEL:
    :param YNUMEL:
    :param ZNUMEL:
    """
    xyzidx = tl.arange(0, XNUMEL * YNUMEL * ZNUMEL)
    XYZmask = xyzidx < (X * Y * Z)
    X = tl.load(x_ptr + xyzidx, mask=XYZmask)
    Y = tl.load(y_ptr + xyzidx, mask=XYZmask)
    ret = tl.cat(X, Y, can_reorder=True)
    oidx = tl.arange(0, XNUMEL * YNUMEL * ZNUMEL * 2)
    tl.store(output_ptr + oidx, ret)


testlist = [
    ('float32', 2, 256, 16),
    ('float32', 8, 8, 4),
    ('float16', 2, 256, 16),
    ('float16', 8, 8, 4),
    ('int8', 2, 256, 16),
    ('int8', 8, 8, 4),
    ('int8', 2, 255, 9),
    ('int16', 3, 5, 3),
    ('int32', 2, 255, 9),
    ('float16', 55, 5, 16),
    ('float16', 4, 5, 17),
    ('float16', 6, 5, 15),
    ('float16', 2, 1928, 3),
    ('float32', 2, 255, 9),
    ('bool', 3, 5, 3),
]


@pytest.mark.parametrize('sigtype, X, Y, Z', testlist)
def test_cat(sigtype, X, Y, Z):
    """
    _summary_

    :param sigtype:
    :param X:
    :param Y:
    :param Z:
    """
    dtype = test_common.get_torch_typename(sigtype)
    XNUMEL = expand_to_next_power_of_two(X)
    YNUMEL = expand_to_next_power_of_two(Y)
    ZNUMEL = expand_to_next_power_of_two(Z)
    x = torch.full((X, Y, Z), 100, dtype=dtype).npu()
    y = torch.full((X, Y, Z), 40, dtype=dtype).npu()
    ans = torch.cat((x, y), dim=0)
    output = torch.randint(1, (X * 2, Y, Z), dtype=dtype).npu()
    fn_npu[1, 1, 1](output, x, y, X, Y, Z, XNUMEL, YNUMEL, ZNUMEL)
    test_common.validate_cmp(sigtype, output, ans)
