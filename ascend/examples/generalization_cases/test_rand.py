import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math


@triton.jit
def kernel_rand(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.rand(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_rand_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.rand(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randn(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_randn_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.randn(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_randint_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.randint(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        mask = (global_offset + indices) < N
        tl.store(x_ptr + global_offset + indices, rand_vals, mask)  # 存储随机数


@triton.jit
def triton_randint4x_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.randint4x(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
def test_case(shape):
    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()

    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    kernel_rand[ncore, 1, 1](y_calf, 10, numel, xblock)
    kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)

    y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    kernel_randint[ncore, 1, 1](y_cali, 10, numel, xblock)
    kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
def test_rand_4d_5d(shape):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_rand_4d_5d[grid](x, *blocks, *blocks, *strides)
    triton_randn_4d_5d[grid](x, *blocks, *blocks, *strides)
    triton_randint_4d_5d[grid](y, *blocks, *blocks, *strides)
    triton_randint4x_4d_5d[grid](y, *blocks, *blocks, *strides)
