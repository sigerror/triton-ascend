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
def atomic_add_broadcast(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    x = tl.load(x_ptr)  # x is scalar or 1D, no mask needed
    
    # Compute y indices
    y_offset = pid * BLOCK_SIZE
    y_indices = y_offset + tl.arange(0, BLOCK_SIZE)
    y_mask = y_indices < n_elements
    
    y_value = tl.load(y_ptr + y_indices, y_mask)
    # Atomic add: y += x (broadcasted)
    tl.atomic_add(out_ptr + y_indices, y_value, mask=y_mask)
    tl.atomic_add(out_ptr + y_indices, x, mask=y_mask)


# 定义不同测试场景的参数组合 (x_shape, y_shape, BLOCK_SIZE)
test_cases = [
    ((1, 1, 1, 1), (1, 1, 1, 4), 4),
    ((1, 1, 1, 3), (1, 5, 1, 3), 5),
    ((3,), (2, 3, 3, 3, 3), 81),
    ((3,), (2, 3, 3, 3), 27),
    ((3,), (2, 3, 3), 9),
    ((3,), (2, 3), 3),
]


def promote_dtype(x_dtype, y_dtype):
    """
    如果 y 的精度低于 x, 则提升 y 的精度以匹配 x。
    """
    # 如果两个数据类型一致，直接返回
    if x_dtype == y_dtype:
        return y_dtype
    
    # 构建类型的优先级列表（从低到高）
    priority = [
        torch.int8, torch.int16, torch.int32,
        torch.float16, torch.bfloat16, torch.float32
    ]

    # 查找两种类型在优先级列表中的位置
    x_priority = priority.index(x_dtype)
    y_priority = priority.index(y_dtype)

    # 如果y的优先级比x小，则提升到x的类型
    if y_priority < x_priority:
        return x_dtype
    else:
        return y_dtype


@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@pytest.mark.parametrize('y_dtype_str', filtered_dtype)
@pytest.mark.parametrize('x_shape, y_shape, BLOCK_SIZE', test_cases)
def test_atomic_add_broadcast_combined(x_dtype_str, y_dtype_str, x_shape, y_shape, BLOCK_SIZE):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    # 先构造 x0
    x0 = torch.full(x_shape, 83.0000, dtype=x_dtype).npu()

    y_raw_dtype = eval('torch.' + y_dtype_str)

    out_dtype = promote_dtype(x_dtype, y_raw_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32

    # 构造y和out
    y = torch.full(y_shape, -105, dtype=y_raw_dtype).npu()
    out = torch.full(y_shape, 0, dtype=out_dtype).npu()

    # 保存副本用于验证
    x_temp = x0.clone()
    y_temp = y.clone()
    out_temp = out.clone()
    
    # 计算网格大小和元素总数
    n_elements = y.numel()
    grid = (n_elements // BLOCK_SIZE,)  # 自动计算需要的线程块数量
    
    # 调用 Triton 核函数
    atomic_add_broadcast[grid](
        x_ptr=x0,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 验证结果：y += x (广播加法)
    expected = out_temp + y_temp + x_temp
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d + TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', filtered_dtype)
def test_atomic_add(dtype, shape):
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full(shape, 2, dtype=eval('torch.' + dtype)).npu()
    y = torch.full(shape, -10, dtype=eval('torch.' + dtype)).npu()

    y_ref = x1 + 0
    x1_ref = x1 + x0
    
    if len(shape) == 2:
        n_elements = shape[0] * shape[1]
        atomic_add[shape[0], 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[1])
    elif len(shape) == 1:
        n_elements = shape[0]
        BLOCK_SIZE = min(1024, shape[0]) # 1024:限制最大线程块大小
        grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE # 向上取整
        atomic_add[grid_size, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
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