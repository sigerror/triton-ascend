# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import triton
import triton.language as tl
import torch
import pytest
import test_common


@triton.jit
def fn_npu(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr,
           ZB: tl.constexpr, outXB: tl.constexpr, outYB: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param x_ptr:
    :param XB:
    :param YB:
    :param ZB:
    :param outXB:
    :param outYB:
    """
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :,
                                               None] * ZB + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    ret = tl.reshape(X, (outXB, outYB))
    oidx = tl.arange(0, outXB)[:, None] * outYB + tl.arange(0, outYB)[None, :]
    tl.store(output_ptr + oidx, ret)


shapelist = [
    (5, 8, 7, 2, 140),
    (27, 8, 2, 72, 6),
    (3, 2, 5, 1, 30),
    (6, 9, 2, 108, 1),
]
typelist = [
    'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'
]


@pytest.mark.parametrize('XB, YB, ZB, outXB, outYB', shapelist)
@pytest.mark.parametrize(
    "sigtype",
    [
        pytest.param(
            sigtype,
            marks=pytest.mark.skipif(sigtype in ["bfloat16", "int64"],
                                     reason="Unsupported for now"),
        ) for sigtype in typelist
    ],
)
def test_reshape(sigtype, XB, YB, ZB, outXB, outYB):
    """
    _summary_

    :param sigtype:
    :param XB:
    :param YB:
    :param ZB:
    :param outXB:
    :param outYB:
    """
    dtype = test_common.get_torch_typename(sigtype)
    x = test_common.generate_tensor((XB, YB, ZB), sigtype).npu()
    result = torch.reshape(x, (outXB, outYB))
    output = torch.randint(1, (outXB, outYB), dtype=dtype).npu()
    fn_npu[1, 1, 1](output, x, XB, YB, ZB, outXB, outYB)
    test_common.validate_cmp(sigtype, output, result)
