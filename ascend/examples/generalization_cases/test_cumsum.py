import math
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime.libentry import libentry

import acc_util
import test_common
from test_common import TestUtils, get_dtype_size


def torch_func(x, dim, reverse):
    if reverse:
        x = torch.flip(x, [dim])
    res = torch.cumsum(x, dim=dim)
    return res


@libentry()
@triton.jit
def triton_kernel_1d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        XBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    idx = tl.arange(0, XBLOCK)
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@libentry()
@triton.jit
def triton_kernel_2d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@libentry()
@triton.jit
def triton_kernel_3d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        numel_z: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
        ZBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    tl.static_assert(
        numel_z == ZBLOCK, "numel_z must be equal to ZBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx_z = tl.arange(0, ZBLOCK)
    idx = idx_x[:, None, None] * numel_r * numel_z + idx_r[None, :, None] * numel_z + idx_z[None, None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@libentry()
@triton.jit
def triton_kernel_4d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        XB: tl.constexpr,
        YB: tl.constexpr,
        ZB: tl.constexpr,
        MB: tl.constexpr,
):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)
    idx = (xidx[:, None, None, None] * YB * ZB * MB + yidx[None, :, None, None] * ZB * MB +
           zidx[None, None, :, None] * MB + midx[None, None, None, :])
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@libentry()
@triton.jit
def triton_kernel_5d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        XB: tl.constexpr,
        YB: tl.constexpr,
        ZB: tl.constexpr,
        MB: tl.constexpr,
        NB: tl.constexpr,
):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)
    nidx = tl.arange(0, NB)
    idx = (xidx[:, None, None, None, None] * YB * ZB * MB * NB + yidx[None, :, None, None, None] * ZB * MB * NB +
           zidx[None, None, :, None, None] * MB * NB + midx[None, None, None, :, None] * NB +
           nidx[None, None, None, None, :])
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def triton_func(x, dim, reverse):
    res = torch.empty_like(x)
    shape = x.size()
    if len(shape) == 1:
        if dim >= 1:
            pytest.skip("dim >= 1 for 1D tensor, skipping.")
        triton_kernel_1d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[0]
        )
    elif len(shape) == 2:
        if dim >= 2:
            pytest.skip("dim >= 2 for 2D tensor, skipping.")
        triton_kernel_2d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[0], x.shape[1]
        )
    elif len(shape) == 3:
        if dim >= 3:
            pytest.skip("dim >= 3 for 3D tensor, skipping.")
        triton_kernel_3d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[0], x.shape[1], x.shape[2]
        )
    elif len(shape) == 4:
        if dim >= 4:
            pytest.skip("dim >= 4 for 4D tensor, skipping.")
        triton_kernel_4d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        )
    elif len(shape) == 5:
        if dim >= 5:
            pytest.skip("dim >= 5 for 5D tensor, skipping.")
        triton_kernel_5d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        )
    else:
        pytest.skip(f"Unsupported tensor dimension: {len(shape)}")

    return res


def cumsum_generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.rand(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=3, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=3, size=shape, dtype=eval('torch.' + dtype))
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# dtype=int8, bool, reverse=True not support;
not_support_dtype = {'int8', 'bool'}
support_dtypes = [dtype for dtype in TestUtils.full_dtype if dtype not in not_support_dtype]


@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("shape", TestUtils.full_shape)
@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("reverse", [False])
def test_cumsum(dtype, shape, dim, reverse):
    if dtype in not_support_dtype:
        pytest.skip("cumsum  does not support int8 and bool.")

    dtype_size = get_dtype_size(dtype)
    if dtype_size * math.prod(shape) >= (TestUtils.ub_size / 4.5):
        pytest.skip(f"dtype:{dtype} shape:{shape} mem overflow")
    x = cumsum_generate_tensor(shape=shape, dtype=dtype)

    x_npu = x.npu()

    triton_res = triton_func(x_npu, dim, reverse)
    torch_res = torch_func(x_npu, dim, reverse)

    x_gold = x.to(torch.float32)
    cpu_res = torch_func(x_gold, dim, reverse)

    if dtype in ('float16', 'bfloat16', 'float32'):
        try:
            print("starting compare of cpu vs triton:")
            acc_util.assert_close(cpu_res, triton_res)
        except Exception as e:
            print(e)
            print("starting compare of cpu vs triton vs torch_npu:")
            acc_util.benchmark_compare_close(cpu_res, triton_res, torch_res)
        print("PASSED")
    elif dtype in support_dtypes:
        test_common.validate_cmp(dtype, triton_res, torch_res)
    else:
        print(f"dtype = {dtype} does not support.")
