# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_multi_return_cross_entropy_case(
    X_ptr,
    Y_ptr,
    X_stride,
    Y_stride,
    n_cols,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    else:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)


@pytest.mark.parametrize('param_list',
                         [
                            [1024, 152064, 11480, 'bfloat16', 'int32', 'float32'],
                         ]
                         )
def test_case(param_list):
    N, M, V, dtype_1, dtype_2, dtype_3 = param_list
    device = 'npu'
    logits_chunk = torch.rand([N, M], device=device, dtype=eval('torch.' + dtype_1))
    target_chunk = torch.rand(N, device=device, dtype=eval('torch.' + dtype_2))
    loss_1d_slice = torch.rand(N, device=device, dtype=eval('torch.' + dtype_3))
    ce_weight = None
    z_loss_1d_slice = None
    softcap = None
    return_z_loss = False
    BLOCK_SIZE = N
    triton_multi_return_cross_entropy_case[(N,)](
        X_ptr=logits_chunk,
        X_stride=logits_chunk.stride(-2),
        Y_ptr=target_chunk,
        Y_stride=target_chunk.stride(-1),
        n_cols=logits_chunk.stride(-2),
        ignore_index=-100,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32)