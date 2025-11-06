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
    block = tl.load_tensor_descriptor(in_desc, [moffset, noffset])
    tl.store_tensor_descriptor(out_desc, [moffset, noffset], block)


# Exercise the functional load/store builtins once to ensure they map through.
@pytest.mark.parametrize("dtype", ["float32"])
def test_tensor_descriptor_functional_interface(dtype):
    """_summary_

    :param dtype: 
    """
    """Copies an entire tensor blockwise using the descriptor builtins."""

    M, N = 32, 128
    inp = test_common.generate_tensor((M, N), dtype).npu()

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp, out)
