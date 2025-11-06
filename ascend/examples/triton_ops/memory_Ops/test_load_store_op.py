# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import triton
import triton.language as tl
import pytest
import test_common


@triton.jit
def triton_load_store(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr,
                      XBLOCK_SUB: tl.constexpr):
    """
    _summary_

    :param in_ptr0:
    :param out_ptr0:
    :param xnumel:
    :param XBLOCK:
    :param XBLOCK_SUB:
    """
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp2 = tmp0
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


testlist = [
    [(2, 4096, 8), 2, 1024],
    [(2, 4096, 9), 6, 256],
    [(8, 8, 4), 2, 64],
    [(3, 8, 4), 3, 4],
]

typelist = [
    'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'
]


@pytest.mark.parametrize('shape, ncore, xblock_sub', testlist)
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
def test_load_store(shape, ncore, xblock_sub, sigtype):
    """
    _summary_

    :param shape:
    :param ncore:
    :param xblock_sub:
    :param sigtype:
    """
    x0 = test_common.generate_tensor(shape, sigtype).npu()
    xblock = int(x0.numel() / ncore)
    y_ref = x0
    y_cal = test_common.generate_tensor(shape, sigtype).npu()
    triton_load_store[ncore, 1, 1](x0, y_cal, x0.numel(), xblock, xblock_sub)
    test_common.validate_cmp(sigtype, y_cal, y_ref)
