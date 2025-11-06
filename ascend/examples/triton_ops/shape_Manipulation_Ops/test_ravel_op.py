# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import triton
import triton.language as tl
import torch
import pytest
import test_common


@triton.jit
def fn_npu_(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr,
            ZB: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param x_ptr:
    :param XB:
    :param YB:
    :param ZB:
    """
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :,
                                               None] * ZB + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    ret = tl.ravel(X)
    oidx = tl.arange(0, XB * YB * ZB)
    tl.store(output_ptr + oidx, ret)


shapelist = [
    (5, 3, 2),
    (27, 3, 2),
    (3, 5, 2),
    (6, 9, 2),
]
typelist = [
    'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'
]


@pytest.mark.parametrize('XB, YB, ZB', shapelist)
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
def test_ravel(sigtype, XB, YB, ZB):
    """
    _summary_

    :param sigtype:
    :param XB:
    :param YB:
    :param ZB:
    """
    dtype = test_common.get_torch_typename(sigtype)
    if sigtype == 'bool':
        x = test_common.generate_tensor((XB, YB, ZB), sigtype).npu()
    else:
        x = torch.randint(low=-128, high=128, size=(XB, YB, ZB),
                          dtype=dtype).npu()
    ans = torch.ravel(x)
    output = torch.randint(1, (XB * YB * ZB, ), dtype=dtype).npu()
    fn_npu_[1, 1, 1](output, x, XB, YB, ZB)
    test_common.validate_cmp(sigtype, output, ans)
