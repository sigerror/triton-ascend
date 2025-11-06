# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import triton
import triton.language as tl
import torch
import pytest
import test_common


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, XB: tl.constexpr, YB: tl.constexpr,
            ZB: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param x_ptr:
    :param y_ptr:
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
    Y = tl.load(y_ptr + idx)
    ret = tl.interleave(X, Y)
    oidx = xidx[:, None, None]*YB*ZB*2+yidx[None, :, None] * \
        ZB*2+tl.arange(0, 2*ZB)[None, None, :]
    tl.store(output_ptr + oidx, ret)


shapelist = [
    (5, 3, 59),
    (61, 3, 7),
    (7, 5, 33),
    (6, 51, 2),
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
def test_interleave(sigtype, XB, YB, ZB):
    """
    _summary_

    :param sigtype:
    :param XB:
    :param YB:
    :param ZB:
    """
    data_type = test_common.get_torch_typename(sigtype)
    x = test_common.generate_tensor((XB, YB, ZB), sigtype).npu()
    y = test_common.generate_tensor((XB, YB, ZB), sigtype).npu()
    output = torch.randint(1, (XB, YB, ZB * 2), dtype=data_type).npu()
    ans = torch.stack((x, y), dim=-1).reshape(XB, YB, ZB * 2)
    fn_npu_[1, 1, 1](output, x, y, XB, YB, ZB)
    test_common.validate_cmp(sigtype, ans, output)
