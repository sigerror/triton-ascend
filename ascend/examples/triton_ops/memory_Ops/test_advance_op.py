# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import triton
import triton.language as tl
import torch
import pytest
import test_common


@triton.jit
def fn_npu(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr,
           ZB: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param x_ptr:
    :param XB:
    :param YB:
    :param ZB:
    """
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(9, 6, 5),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    bbptr = tl.advance(block_ptr_in, (-9, -6, -5))
    X = tl.load(bbptr)
    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    tl.store(block_ptr_out, X)


shapelist = [
    (5, 8, 7),
    (27, 8, 2),
    (3, 2, 5),
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
            marks=pytest.mark.skipif(sigtype in ["bfloat16", "bool", "int64"],
                                     reason="Unsupported for now"),
        ) for sigtype in typelist
    ],
)
def test_advance(sigtype, XB, YB, ZB):
    """
    _summary_

    :param sigtype:
    :param XB:
    :param YB:
    :param ZB:
    """
    dtype = test_common.get_torch_typename(sigtype)
    x = test_common.generate_tensor((XB, YB, ZB), sigtype).npu()

    output = torch.randint(1, (XB, YB, ZB), dtype=dtype).npu()
    a = x
    fn_npu[1, 1, 1](output, x, XB=XB, YB=YB, ZB=ZB)
    torch.testing.assert_close(output, a)
