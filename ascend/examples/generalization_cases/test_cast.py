# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import math
import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils, check_ub_mem_overflow, get_dtype_size

@triton.jit
def cast_to_bool(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int1)
    tl.store(output_ptr + idx, ret)

@triton.jit
def cast_to_i8(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int8)
    tl.store(output_ptr + idx, ret)

@triton.jit
def cast_to_i16(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int16)
    tl.store(output_ptr + idx, ret)

@triton.jit
def cast_to_i32(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int32)
    tl.store(output_ptr + idx, ret)

@triton.jit
def cast_to_i64(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int64)
    tl.store(output_ptr + idx, ret)

@triton.jit
def cast_to_fp32(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.float32)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_fp16(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.float16)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_bf16(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.bfloat16)
    tl.store(output_ptr + idx, ret)

@triton.jit
def cast_to_uint32(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.uint32)
    tl.store(output_ptr + idx, ret)

@triton.jit
def cast_to_int64(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int64)
    tl.store(output_ptr + idx, ret)

triton_func_map = {
    "bool": cast_to_bool,
    "int8": cast_to_i8,
    "int16": cast_to_i16,
    "int32": cast_to_i32,
    "float16": cast_to_fp16,
    "bfloat16": cast_to_bf16,
    "float32": cast_to_fp32,
    "uint32": cast_to_uint32,
    "int64": cast_to_int64
}

def structParam(x0):
    dim = x0.dim()
    stride0, stride1, stride2 = 0, 0, 0
    shape0, shape1, shape2 = 0, 0, 0
    if dim >= 1:
        stride0 = x0.stride(0)
        shape0 = x0.shape[0]
    if dim >= 2:
        stride1 = x0.stride(1)
        shape1 = x0.shape[1]
    if dim == 3:
        stride2 = x0.stride(2)
        shape2 = x0.shape[2]
    return dim, stride0, stride1, stride2, shape0, shape1, shape2



@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('srcDtype', TestUtils.full_dtype)
@pytest.mark.parametrize('dstDtype', TestUtils.full_dtype)
def test_cast(srcDtype, dstDtype, shape):
    if srcDtype == dstDtype:
        return
    srcBytes = get_dtype_size(srcDtype)
    dstBytes = get_dtype_size(dstDtype)
    dtype_size = max(srcBytes, dstBytes)
    if dstDtype == 'int8':
        if dtype_size * math.prod(shape) >= (TestUtils.ub_size / 100):
            print(f"srcDtype:{srcDtype}, dstDtype:{dstDtype}, shape:{shape} mem overflow")
            return
    elif dtype_size * math.prod(shape) >= (TestUtils.ub_size / 12):
        print(f"srcDtype:{srcDtype}, dstDtype:{dstDtype}, shape:{shape} mem overflow")
        return
    
    x0 = test_common.generate_tensor(shape, srcDtype)
    torch_res = x0.to(eval("torch." + dstDtype))
    x0 = x0.npu()
    triton_func = triton_func_map.get(dstDtype, None)
    assert triton_func is not None, f"triton_func not Found, srcDtype:{srcDtype}, dstDtype:{dstDtype}"
    triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype)).npu()
    dim, stride0, stride1, stride2, XB, YB, ZB = structParam(x0)
    assert 0 <= dim <= 3, f"dim out of range [0, 3], dim:{dim}"
    triton_func[1, 1, 1](triton_res, x0, stride0, stride1, stride2, dim, XB, YB, ZB)
    test_common.validate_cmp(dstDtype, triton_res, torch_res)


@triton.jit
def cast_to_multi_d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
    dtype = output_ptr.type.element_ty

    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    if (YB * ZB * MB * NB) > 1:
        offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    if (ZB * MB * NB) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    if (MB * NB) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    if NB > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    X = tl.load(x_ptr + offsets)
    ret = tl.cast(X, dtype=dtype)

    tl.store(output_ptr + offsets, ret)


def cast_npu_multi_d(para_type, data_type, to_para, to_dtype, XB, YB, ZB, MB, NB):
    print(f"TESTING: cast from {para_type} to {to_para} in shape ({XB}, {YB}, {ZB}, {MB}, {NB})")

    if para_type == '*i1':
        x = torch.randint(low=0, high=2, size=(XB, YB, ZB, MB, NB), dtype=data_type).npu()
    elif para_type == '*i8' or para_type == '*i16' or para_type == '*i32' or para_type == '*64':
        x = torch.randint(low=-128, high=128, size=(XB, YB, ZB, MB, NB), dtype=data_type).npu()
    elif para_type == '*i16':
        x = torch.randint(low=-32768, high=32768, size=(XB, YB, ZB, MB, NB), dtype=data_type).npu()
    elif para_type == '*i32':
        x = torch.randint(low=-65536, high=65536, size=(XB, YB, ZB, MB, NB), dtype=data_type).npu()
    elif para_type == '*i64':
        x = torch.randint(low=-65536, high=65536, size=(XB, YB, ZB, MB, NB), dtype=data_type).npu()
    else:  # float
        x = torch.randn((XB, YB, ZB, MB, NB), dtype=data_type).npu()

    if to_para == '*i1':
        cmp_type = "bool"
    elif to_para == '*i8':
        cmp_type = "int8"
    elif to_para == '*i16':
        cmp_type = "int16"
    elif to_para == '*i32':
        cmp_type = "int32"
    elif to_para == '*i64':
        cmp_type = "int64"
    elif to_para == '*fp16':
        cmp_type = "float16"
    elif to_para == '*fp32':
        cmp_type = "float32"
    elif to_para == '*bf16':
        cmp_type = "bfloat16"

    output = torch.randint(1, (XB, YB, ZB, MB, NB), dtype=to_dtype).npu()

    a = x.to(to_dtype)

    cast_to_multi_d[(1, )](output, x, XB, YB, ZB, MB, NB)

    test_common.validate_cmp(cmp_type, a, output)


@pytest.mark.shape_4d_5d
def test_cast_high_priority_dtype_4d_5d():
    typelist = [
        (torch.int8, '*i8'),
        (torch.float32, '*fp32'),
        (torch.float16, '*fp16'),
    ]

    shapes = [(4, 6, 2, 4, 2)]
    ContinueList = []
    for src in typelist:
        for dst in typelist:
            if src != dst and (src[1], dst[1]) not in ContinueList:
                for shape in shapes:
                    cast_npu_multi_d(src[1], src[0], dst[1], dst[0], shape[0], shape[1], shape[2], shape[3], shape[4])

    print("test_cast_full_multi_d passed")


if __name__ == "__main__":
    for shape in [(3, ), (3, 3), (3, 3, 3)]:
        for srcDtype in ['int8', 'float32', 'bool']:
            for dstDtype in ['int8', 'float32', 'bool']:
                test_cast(srcDtype, dstDtype, shape)
