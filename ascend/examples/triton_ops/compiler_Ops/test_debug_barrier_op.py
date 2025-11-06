# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pytest
import triton
import triton.language as tl
import test_common


@triton.jit
def triton_debug_barrier(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr,
                         XBLOCK_SUB: tl.constexpr):
    """_summary_

    :param in_ptr0: 
    :param out_ptr0: 
    :param xnumel: 
    :param XBLOCK: 
    :param XBLOCK_SUB: 
    """
    xoffset = tl.program_id(0) * XBLOCK
    tl.multiple_of(in_ptr0, 32)
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tl.max_contiguous(tmp0, XBLOCK_SUB)
        tl.max_constancy(tmp0, XBLOCK_SUB)
        tmp2 = tmp0
        tl.debug_barrier()
        tl.store(out_ptr0 + x_index, tmp2, xmask)


typelist = [
    'bool', 'int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32'
]


@pytest.mark.parametrize('param_list', [
    [(2, 4096, 8), 2, 32768, 1024],
    [(8, 8, 4), 2, 128, 64],
])
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
def test_debug_barrier(param_list, sigtype):
    """_summary_

    :param param_list:
    :param sigtype:
    """
    shape, ncore, xblock, xblock_sub = param_list
    dtype = sigtype
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x0
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_debug_barrier[ncore, 1, 1](x0, y_cal, x0.numel(), xblock,
                                      xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)
