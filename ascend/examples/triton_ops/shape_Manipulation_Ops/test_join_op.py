# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import triton
import triton.language as tl
import torch
import pytest
import test_common


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, XB: tl.constexpr, YB: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param x_ptr:
    :param y_ptr:
    :param XB:
    :param YB:
    """
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    idx = xidx[:, None] * YB + yidx[None, :]
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)
    ret = tl.join(X, Y)
    oidx = xidx[:, None, None]*YB*2+yidx[None, :, None] * \
        2+tl.arange(0, 2)[None, None, :]
    tl.store(output_ptr + oidx, ret)


shapelist = [
    (5, 3),
    (27, 3),
    (3, 5),
    (6, 9),
]
typelist = [
    'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'
]


@pytest.mark.parametrize('XB, YB', shapelist)
@pytest.mark.parametrize(
    "sigtype",
    [
        pytest.param(
            sigtype,
            marks=pytest.mark.skipif(sigtype in ["bfloat16", "bool", "int64"],
                                     reason="Unsupported for now"),
        ) for sigtype in typelist
    ],
)
def test_join(sigtype, XB, YB):
    """
    _summary_

    :param sigtype:
    :param XB:
    :param YB:
    """
    data_type = test_common.get_torch_typename(sigtype)
    x = torch.full((XB, YB), 100, dtype=data_type).npu()
    y = torch.full((XB, YB), 30, dtype=data_type).npu()
    ans = torch.stack((x, y), dim=-1)
    output = torch.randint(1, (XB, YB, 2), dtype=data_type).npu()
    fn_npu_[1, 1, 1](output, x, y, XB, YB)
    test_common.validate_cmp(sigtype, ans, output)
