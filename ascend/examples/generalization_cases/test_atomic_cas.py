import math
import pytest
import torch
import triton

import triton.language as tl

import test_common
from test_common import TestUtils
filtered_dtype = [dtype for dtype in TestUtils.full_dtype if dtype not in {'uint32', 'bfloat16', 'int8', 'bool'}]


@triton.jit
def atomic_cas(in_ptr0, in_ptr1, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr, BLOCK_NUM: tl.constexpr):
    in_offset = tl.program_id(0) * BLOCK_SIZE
    out_offset = (tl.program_id(0) % BLOCK_NUM) * BLOCK_SIZE
    in_index = in_offset + tl.arange(0, BLOCK_SIZE)
    out_index = out_offset + tl.arange(0, BLOCK_SIZE)
    xmask = in_index < n_elements

    tmp0 = tl.load(in_ptr0 + (in_index), xmask)
    tmp1 = tl.load(in_ptr1 + (in_index), xmask)
    tl.atomic_cas(out_ptr0 + (out_index), tmp1, tmp0)



@triton.jit
def atomic_cas_broadcast(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    x = tl.load(x_ptr)  # x is scalar or 1D, no mask needed
    
    # Compute y indices
    y_offset = pid * BLOCK_SIZE
    y_indices = y_offset + tl.arange(0, BLOCK_SIZE)
    y_mask = y_indices < n_elements
    
    y_value = tl.load(y_ptr + y_indices, y_mask)
    # Atomic or: y |= x (broadcasted)
    tl.atomic_cas(out_ptr + y_indices, y_value, mask=y_mask)
    tl.atomic_cas(out_ptr + y_indices, x, mask=y_mask)


# 定义不同测试场景的参数组合 (x_shape, y_shape, BLOCK_SIZE)
test_cases = [
    ((1, 1, 1, 1), (1, 1, 1, 4), 4),
    ((1, 1, 1, 3), (1, 5, 1, 3), 5),
    ((3,), (2, 3, 3, 3, 3), 81),
    ((3,), (2, 3, 3, 3), 27),
    ((3,), (2, 3, 3), 9),
    ((3,), (2, 3), 3),
]


@pytest.mark.parametrize('shape', TestUtils.test_shape2d + TestUtils.test_shape1d)
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_cas(x_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)

    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = test_common.generate_tensor(x_shape, x_dtype_str).npu()
    c = torch.randint(low=0, high=2, size=x_shape, dtype=x_dtype).npu()
    y = torch.randint(low=0, high=2, size=shape, dtype=x_dtype).npu()

    # 保存副本用于验证
    x_temp = x.clone()
    c_temp = c.clone()
    y_temp = y.clone()
    
    if len(shape) == 2:
        n_elements = shape[0] * shape[1] * 2
        atomic_cas[shape[0] * 2, 1, 1](x, c, y, n_elements, BLOCK_SIZE=shape[1], BLOCK_NUM=shape[0])
    elif len(shape) == 1:
        n_elements = shape[0]
        BLOCK_SIZE = min(1024, shape[0]) # 1024:限制最大线程块大小
        grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE # 向上取整
        aligned_size = grid_size * BLOCK_SIZE
        # value
        x_concat = torch.full([aligned_size * 2], 0, dtype=x_dtype).npu()
        x_concat[0:n_elements] = x[0:n_elements]
        x_concat[aligned_size:(aligned_size + n_elements)] = x[n_elements:(n_elements * 2)]
        # compare
        c_concat = torch.full([aligned_size * 2], 0, dtype=x_dtype).npu()
        c_concat[0:n_elements] = c[0:n_elements]
        c_concat[aligned_size:(aligned_size + n_elements)] = c[n_elements:(n_elements * 2)]
        atomic_cas[grid_size * 2, 1, 1](x_concat, c_concat, y, aligned_size * 2, BLOCK_SIZE=BLOCK_SIZE, BLOCK_NUM=grid_size)
    
    expected = torch.where(y_temp == c_temp[0:shape[0]], x_temp[0:shape[0]], y_temp)
    expected = torch.where(expected == c_temp[shape[0]:(shape[0] * 2)], x_temp[shape[0]:(shape[0] * 2)], expected)
    torch.testing.assert_close(y, expected)


# 3d
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_cas_3d(x_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)

    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = test_common.generate_tensor(x_shape, x_dtype_str).npu()
    c = torch.randint(low=3, high=5, size=x_shape, dtype=x_dtype).npu()
    y = torch.randint(low=3, high=5, size=shape, dtype=x_dtype).npu()

    # 保存副本用于验证
    x_temp = x.clone()
    c_temp = c.clone()
    y_temp = y.clone()

    n_elements = shape[0] * shape[1] * shape[2]
    atomic_cas[2, 1, 1](x, c, y, n_elements * 2, BLOCK_SIZE=shape[0] * shape[1] * shape[2], BLOCK_NUM=1)

    expected = torch.where(y_temp == c_temp[0:shape[0]], x_temp[0:shape[0]], y_temp)
    expected = torch.where(expected == c_temp[shape[0]:(shape[0] * 2)], x_temp[shape[0]:(shape[0] * 2)], expected)
    torch.testing.assert_close(y, expected)


@triton.jit
def atomic_cas_multi_d(in_ptr0, in_ptr1, out_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + offsets)
    tl.atomic_cas(out_ptr0 + offsets, tmp1, tmp0)


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
def test_atomic_cas_4d_5d(dtype, shape):
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    c = torch.randint(low=2, high=4, size=shape, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.randint(low=2, high=4, size=shape, dtype=eval('torch.' + dtype)).npu()

    x1_ref = torch.where(x1 == c, 3, x1)

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)

    atomic_cas_multi_d[(1, )](x0, c, x1, *triton_shape)
    test_common.validate_cmp(dtype, x1, x1_ref)


@triton.jit
def atomic_cas_5d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr, MB1: tl.constexpr, NB1: tl.constexpr):
    base = tl.program_id(0) * (XB * YB * ZB * MB * NB)
    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1 * MB1 * NB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1 * MB1 * NB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :] * (MB1 * NB1)
    offsets1 = offsets1[:, :, :, None] + tl.arange(0, MB1)[None, None, None, :] * NB1
    offsets1 = offsets1[:, :, :, :, None] + tl.arange(0, NB1)[None, None, None, None, :]
    
    based_offsets = offsets + base
    
    tmp0 = tl.load(x_ptr + based_offsets)
    tmp1 = tl.load(y_ptr + based_offsets)
    tl.atomic_cas(out_ptr + offsets1, tmp1, tmp0)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 2, 1, 1), (1, 1, 2, 1, 1)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_cas_5d(x_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(x0_shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()

    c = torch.randint(low=3, high=5, size=x_shape, dtype=x_dtype).npu()
    out = torch.randint(low=3, high=5, size=y_shape, dtype=x_dtype).npu()

    x_temp = x.clone()
    c_temp = c.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    XB, YB, ZB, MB, NB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 5:
        triton_shape1.append(1)
    XB1, YB1, ZB1, MB1, NB1 = triton_shape1

    atomic_cas_5d[(2, )](
        x_ptr=x,
        y_ptr=c,
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB, MB=MB, NB=NB,
        XB1=XB1, YB1=YB1, ZB1=ZB1, MB1=MB1, NB1=NB1,
        )
    
    expected = torch.where(out_temp == c_temp[0:x0_shape[0]], x_temp[0:x0_shape[0]], out_temp)
    expected = torch.where(expected == c_temp[x0_shape[0]:(x_shape[0])], x_temp[x0_shape[0]:(x_shape[0])], expected)
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_cas_4d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr, MB1: tl.constexpr):
    base = tl.program_id(0) * (XB * YB * ZB * MB)
    offsets = tl.arange(0, XB) * (YB * ZB * MB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB)
    offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] 

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1 * MB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1 * MB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :] * (MB1)
    offsets1 = offsets1[:, :, :, None] + tl.arange(0, MB1)[None, None, None, :]
    
    based_offsets = offsets + base

    tmp0 = tl.load(x_ptr + based_offsets)
    tmp1 = tl.load(y_ptr + based_offsets)
    tl.atomic_cas(out_ptr + offsets1, tmp1, tmp0)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 2, 1), (1, 1, 2, 1)],
        [(1, 1, 2, 2), (1, 1, 2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_cas_4d(x_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(x0_shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    c = torch.randint(low=3, high=5, size=x_shape, dtype=x_dtype).npu()
    out = torch.randint(low=3, high=5, size=y_shape, dtype=x_dtype).npu()

    x_temp = x.clone()
    c_temp = c.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 4:
        triton_shape.append(1)
    XB, YB, ZB, MB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 4:
        triton_shape1.append(1)
    XB1, YB1, ZB1, MB1 = triton_shape1

    atomic_cas_4d[(2, )](
        x_ptr=x,
        y_ptr=c,
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB, MB=MB,
        XB1=XB1, YB1=YB1, ZB1=ZB1, MB1=MB1,
        )
    
    expected = torch.where(out_temp == c_temp[0:x0_shape[0]], x_temp[0:x0_shape[0]], out_temp)
    expected = torch.where(expected == c_temp[x0_shape[0]:(x_shape[0])], x_temp[x0_shape[0]:(x_shape[0])], expected)
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_cas_3d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr):
    base = tl.program_id(0) * (XB * YB * ZB)
    offsets = tl.arange(0, XB) * (YB * ZB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :]

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :]
    
    based_offsets = offsets + base

    tmp0 = tl.load(x_ptr + based_offsets)
    tmp1 = tl.load(y_ptr + based_offsets)
    tl.atomic_cas(out_ptr + offsets1, tmp1, tmp0)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 2), (1, 1, 2)],
        [(1, 2, 2), (1, 2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_cas_3d_2(x_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(x0_shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    c = torch.randint(low=3, high=5, size=x_shape, dtype=x_dtype).npu()
    out = torch.randint(low=3, high=5, size=y_shape, dtype=x_dtype).npu()

    x_temp = x.clone()
    c_temp = c.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 3:
        triton_shape.append(1)
    XB, YB, ZB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 3:
        triton_shape1.append(1)
    XB1, YB1, ZB1 = triton_shape1

    atomic_cas_3d[(2, )](
        x_ptr=x,
        y_ptr=c,
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB,
        XB1=XB1, YB1=YB1, ZB1=ZB1,
        )
    
    expected = torch.where(out_temp == c_temp[0:x0_shape[0]], x_temp[0:x0_shape[0]], out_temp)
    expected = torch.where(expected == c_temp[x0_shape[0]:(x_shape[0])], x_temp[x0_shape[0]:(x_shape[0])], expected)
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_cas_2d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr):
    base = tl.program_id(0) * (XB * YB)
    offsets = tl.arange(0, XB) * (YB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :]

    offsets1 = tl.arange(0, XB1) * (YB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :]
    
    based_offsets = offsets + base

    tmp0 = tl.load(x_ptr + based_offsets)
    tmp1 = tl.load(y_ptr + based_offsets)
    tl.atomic_cas(out_ptr + offsets1, tmp1, tmp0)


@pytest.mark.parametrize('param_list',
    [
        [(1, 2), (1, 2)],
         [(2, 2), (2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_cas_2d(x_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(x0_shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    c = torch.randint(low=3, high=5, size=x_shape, dtype=x_dtype).npu()
    out = torch.randint(low=3, high=5, size=y_shape, dtype=x_dtype).npu()

    x_temp = x.clone()
    c_temp = c.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 2:
        triton_shape.append(1)
    XB, YB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 2:
        triton_shape1.append(1)
    XB1, YB1 = triton_shape1

    atomic_cas_2d[(2, )](
        x_ptr=x,
        y_ptr=c,
        out_ptr=out, 
        XB=XB, YB=YB,
        XB1=XB1, YB1=YB1,
        )
    
    expected = torch.where(out_temp == c_temp[0:x0_shape[0]], x_temp[0:x0_shape[0]], out_temp)
    expected = torch.where(expected == c_temp[x0_shape[0]:(x_shape[0])], x_temp[x0_shape[0]:(x_shape[0])], expected)
    torch.testing.assert_close(out, expected)