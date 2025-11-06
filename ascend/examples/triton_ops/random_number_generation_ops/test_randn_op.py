# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
import triton
import triton.language as tl
import torch
import pytest


@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr,
                 XBLOCK: tl.constexpr):
    """_summary_

    :param x_ptr: 
    :param n_rounds: 
    :param N: 
    :param XBLOCK: 
    """
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randn(5, 10 + global_offset, n_rounds)  # generate a random number for each index
        tl.store(x_ptr + global_offset, rand_vals)  # store random number


shapes = [(1, 3)]


@pytest.mark.parametrize('shape', shapes)
def test_randn(shape):
    """_summary_

    :param shape: 
    """
    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()

    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)
