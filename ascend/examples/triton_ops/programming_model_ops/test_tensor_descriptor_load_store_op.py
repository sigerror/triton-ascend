# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pytest
import torch
import triton
import triton.language as tl
import test_common


@triton.jit
def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    """_summary_

    :param out_ptr: 
    :param a_ptr: 
    :param M: 
    :param N: 
    :param M_BLOCK: 
    :param N_BLOCK: 
    """
    in_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    block = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], block)


typelist = [
    'float32', 'float16', 'bfloat16', 'int32', 'int64', 'int16', 'int8'
]


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
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16)])
def test_tensor_descriptor_load_store(sigtype, M_BLOCK, N_BLOCK):
    """_summary_

    :param sigtype: 
    :param M_BLOCK: 
    :param N_BLOCK: 
    """
    dtype = sigtype
    M, N = M_BLOCK * 2, N_BLOCK * 2
    inp = test_common.generate_tensor((M, N), dtype).npu()
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp, out)
