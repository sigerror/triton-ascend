import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils


@triton.jit
def triton_test_fn_atomic_min_dma(in_ptr0, out_ptr0, out_ptr1, n_elements : tl.constexpr, BLOCK_SIZE : tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_min(out_ptr0 + (x1), tmp0, xmask)


# torch.min do not support int
@pytest.mark.parametrize('shape', TestUtils.test_shape2d + TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
def test_atomic_min(dtype, shape):
    x0 = test_common.generate_tensor(shape, dtype)
    x1 = test_common.generate_tensor(shape, dtype)
    y = test_common.generate_tensor(shape, dtype)

    x1_ref = torch.minimum(x0, x1)
    x0 = x0.npu()
    x1 = x1.npu()
    y = y.npu()

    if len(shape) == 2:
        n_elements = shape[0] * shape[1]
        triton_test_fn_atomic_min_dma[shape[0], 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[1])
    elif len(shape) == 1:
        n_elements = shape[0]
        BLOCK_SIZE = min(1024, shape[0]) # 1024:限制最大线程块大小
        grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE # 向上取整
        triton_test_fn_atomic_min_dma[grid_size, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    test_common.validate_cmp(dtype, x1, x1_ref)


# 3d
testlist = [
    (1,22,39),
    (27,1,39),
    (27,22,1),
    (1,1,23),
    (23,1,1),
    (1,23,1),
    (27,5,3),
    (2,29,4),
    (7,31,7),
    (3,5,8),
    (7,17,15),
    (25,5,16),
    (23,5,31),
    (7,11,32),
    (7,11,33),
    (2,3,255),
    (3,3,256),
    (3,2,257),
]


@pytest.mark.parametrize('shape', testlist)
@pytest.mark.parametrize('dtype', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
def test_atomic_min_3d(dtype, shape):
    ncore = 1
    # block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    # old size: (32768, 256)
    # tensor of (1024, 256) is too big, and it will lead to failure in the backend
    # so here we make it smaller
    x0 = test_common.generate_tensor(shape, dtype)
    x1 = test_common.generate_tensor(shape, dtype)
    y = test_common.generate_tensor(shape, dtype)

    x1_ref = torch.minimum(x0, x1)
    x0 = x0.npu()
    x1 = x1.npu()
    y = y.npu()

    n_elements = shape[0] * shape[1] * shape[2]
    triton_test_fn_atomic_min_dma[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1] * shape[2])
    test_common.validate_cmp(dtype, x1, x1_ref)


@triton.jit
def atomic_min_multi_d(in_ptr0, out_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
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
    tl.atomic_min(out_ptr0 + offsets, tmp0)


filtered_dtype = [dtype for dtype in TestUtils.full_dtype if dtype not in {'int64', 'bool'}]


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
def test_atomic_min_4d_5d(dtype, shape):
    x0_value = 1
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full(shape, 2, dtype=eval('torch.' + dtype)).npu()

    x1_ref = torch.minimum(x1, x0)

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    atomic_min_multi_d[(1, )](x0, x1, *triton_shape)
    test_common.validate_cmp(dtype, x1, x1_ref)
