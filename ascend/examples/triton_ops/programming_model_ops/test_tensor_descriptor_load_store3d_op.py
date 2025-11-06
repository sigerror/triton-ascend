# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pytest
import torch
import triton
import triton.language as tl
import test_common


@triton.jit
def kernel(out_ptr, a_ptr, M, N, K, stride_m, stride_n, stride_k,
           M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
           K_BLOCK: tl.constexpr):
    """_summary_

    :param out_ptr: 
    :param a_ptr: 
    :param M: 
    :param N: 
    :param K: 
    :param stride_m: 
    :param stride_n: 
    :param stride_k: 
    :param M_BLOCK: 
    :param N_BLOCK: 
    :param K_BLOCK: 
    """
    in_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N, K],
        strides=[stride_m, stride_n, stride_k],
        block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N, K],
        strides=[stride_m, stride_n, stride_k],
        block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    koffset = tl.program_id(2) * K_BLOCK
    block = in_desc.load([moffset, noffset, koffset])
    out_desc.store([moffset, noffset, koffset], block)


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
def test_tensor_descriptor_load_store3d(sigtype):
    """_summary_

    :param sigtype: 
    """

    dtype = sigtype
    M, N, K = 8, 16, 32
    inp = test_common.generate_tensor((M, N, K), dtype).npu()
    out = inp.new_empty((M, N, K))

    M_BLOCK = 2
    N_BLOCK = 4

    # automately adjust K_BLOCK，guarantee the last dimension of block is at least 16 bytes
    dtype = getattr(inp, "dtype", None)
    itemsize = torch.tensor([], dtype=inp.dtype).element_size()
    min_k_block = max(16 // itemsize, 1)
    K_BLOCK = max(8, min_k_block)

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK
    grid_k = K // K_BLOCK

    kernel[(grid_m, grid_n, grid_k)](out, inp, *inp.shape, *out.stride(),
                                     M_BLOCK, N_BLOCK, K_BLOCK)
    torch.testing.assert_close(inp.reshape(M * N * K), out.reshape(M * N * K))
