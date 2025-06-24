import math
import pytest
import torch
import triton

import triton.language as tl

import test_common
from test_common import TestUtils
filtered_dtype = [dtype for dtype in TestUtils.full_dtype if dtype not in {'int64', 'bool'}]


@triton.jit
def atomic_add(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_add(out_ptr0 + (x1), tmp0, xmask)


@triton.jit
def atomic_add_broadcast(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    x = tl.load(x_ptr)  # x is scalar or 1D, no mask needed
    
    # Compute y indices
    y_offset = pid * BLOCK_SIZE
    y_indices = y_offset + tl.arange(0, BLOCK_SIZE)
    y_mask = y_indices < n_elements
    
    # Atomic add: y += x (broadcasted)
    tl.atomic_add(y_ptr + y_indices, x, mask=y_mask)


# 定义不同测试场景的参数组合 (x_shape, y_shape, BLOCK_SIZE)
test_cases = [
    ((3,), (2, 3, 3, 3, 3), 81),
    ((3,), (2, 3, 3, 3), 27),
    ((3,), (2, 3, 3), 9),
    ((3,), (2, 3), 3),
]


@pytest.mark.parametrize('dtype', filtered_dtype)
@pytest.mark.parametrize('x_shape, y_shape, BLOCK_SIZE', test_cases)
def test_atomic_add_broadcast_combined(dtype, x_shape, y_shape, BLOCK_SIZE):
    # 初始化输入张量
    x0 = torch.full(x_shape, 3, dtype=eval('torch.' + dtype)).npu()
    y = torch.full(y_shape, 2, dtype=eval('torch.' + dtype)).npu()
    y_temp = y.clone()  # 保存原始值用于验证
    
    # 计算网格大小和元素总数
    n_elements = y.numel()
    grid = (n_elements // BLOCK_SIZE,)  # 自动计算需要的线程块数量
    
    # 调用 Triton 核函数
    atomic_add_broadcast[grid](
        x_ptr=x0,
        y_ptr=y,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 验证结果：y += x (广播加法)
    expected = y_temp + 3
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', filtered_dtype)
def test_atomic_add(dtype, shape):
    ncore = 1
    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full((split_size, shape[1]), 2, dtype=eval('torch.' + dtype)).npu()
    y = torch.full((split_size, shape[1]), -10, dtype=eval('torch.' + dtype)).npu()

    y_ref = x1 + 0
    x1_ref = x1 + ncore * x0_value

    n_elements = shape[0] * shape[1]
    atomic_add[shape[0], 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[1])
    test_common.validate_cmp(dtype, x1, x1_ref)


# 3d
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
@pytest.mark.parametrize('dtype', filtered_dtype)
def test_atomic_add_3d(dtype, shape):
    ncore = 1
    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full((split_size, shape[1], shape[2]), 2, dtype=eval('torch.' + dtype)).npu()
    y = torch.full((split_size, shape[1], shape[2]), -10, dtype=eval('torch.' + dtype)).npu()

    x1_ref = x1 + ncore * x0_value

    n_elements = shape[0] * shape[1] * shape[2]
    atomic_add[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[0] * shape[1] * shape[2])
    test_common.validate_cmp(dtype, x1, x1_ref)


@triton.jit
def atomic_add_multi_d(in_ptr0, out_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    if (YB * ZB * MB * NB) > 1:
        offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    if (ZB * MB * NB) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    if (MB * NB) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    if NB > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]
    
    tmp0 = tl.load(in_ptr0 + offsets)
    tl.atomic_add(out_ptr0 + offsets, tmp0)


# multi_d
@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('shape', [
    (2, 4, 8, 4),
    (8, 4, 2, 4),
    (2, 8, 2, 2),
    (2, 4, 8, 4, 2),
    (8, 4, 2, 4, 4),
    (2, 8, 2, 2, 2),
])
@pytest.mark.parametrize('dtype', filtered_dtype)
def test_atomic_add_4d_5d(dtype, shape):
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full(shape, 2, dtype=eval('torch.' + dtype)).npu()

    x1_ref = x1 + x0_value

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    atomic_add_multi_d[(1, )](x0, x1, *triton_shape)
    test_common.validate_cmp(dtype, x1, x1_ref)