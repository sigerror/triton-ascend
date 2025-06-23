# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils
import logging

@triton.jit
def fn_npu_1d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr):
    yidx = tl.arange(0, YB)

    X = tl.load(x_ptr + yidx)

    ret = tl.expand_dims(X, 1)

    oidx = yidx[:, None] + tl.arange(0, 1)[None, :]

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_expand_dims_1d(shape, dtype):
    x = test_common.generate_tensor(shape,dtype).npu()
    a = x.unsqueeze(1)

    output = torch.randint(1, (shape[0], 1), dtype=eval('torch.' + dtype)).npu()

    fn_npu_1d[1, 1, 1](output, x, YB=shape[0], ZB=1, debug=True)

    torch.testing.assert_close(output, a)


@triton.jit
def fn_npu_2d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr):
    yoffs = tl.program_id(0)
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB)

    idx = yidx[:, None] * ZB + zidx[None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.expand_dims(X, 1)

    oidx = yidx[:, None, None] * ZB + tl.arange(0, 1)[None, :, None] + zidx[None, None, :]

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_expand_dims_2d(shape, dtype):
    x = test_common.generate_tensor(shape,dtype).npu()
    a = x.unsqueeze(1)

    output = torch.randint(1, (shape[0], 1, shape[1]), dtype=eval('torch.' + dtype)).npu()

    if x.numel()*x.element_size()>8192:
        fn_npu_2d[shape[0],1 ,1](output, x, YB=1, ZB=shape[1])
    else:
        fn_npu_2d[1, 1, 1](output, x, YB=shape[0], ZB=shape[1])

    torch.testing.assert_close(output, a)


@triton.jit
def fn_npu_3d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.expand_dims(X, 2)

    oidx = xidx[:, None, None, None] * YB * ZB + yidx[None, :, None, None] * ZB + tl.arange(0, 1)[None, None, :,
                                                                                  None] + zidx[None, None, None, :]

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
def test_expand_dims_3d(dtype, shape):
    x = test_common.generate_tensor(shape,dtype).npu()
    a = x.unsqueeze(2)

    output = torch.randint(1, (shape[0], shape[1], 1, shape[2]), dtype=eval('torch.' + dtype)).npu()


    fn_npu_3d[1, 1, 1](output, x, XB=shape[0], YB=shape[1], ZB=shape[2])

    torch.testing.assert_close(output, a)


@triton.jit
def fn_npu_4d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)

    idx = xidx[:, None, None, None] * YB * ZB * MB + yidx[None, :, None, None] * ZB * MB + zidx[None, None, :, None] * MB + midx[None, None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.expand_dims(X, 2)

    oidx = xidx[:, None, None, None, None] * YB * ZB * MB + yidx[None, :, None, None, None] * ZB * MB + tl.arange(0, 1)[None, None, :, None, None] + zidx[None, None, None, :, None] * MB + midx[None, None, None, None, :]

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_5d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)
    nidx = tl.arange(0, NB)

    idx = xidx[:, None, None, None, None] * YB * ZB * MB * NB + yidx[None, :, None, None, None] * ZB * MB * NB + zidx[None, None, :, None, None] * MB * NB + midx[None, None, None, :, None] * NB + nidx[None, None, None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.expand_dims(X, 2)

    oidx = xidx[:, None, None, None, None, None] * YB * ZB * MB * NB + yidx[None, :, None, None, None, None] * ZB * MB * NB + tl.arange(0, 1)[None, None, :, None, None, None] + zidx[None, None, None, :, None, None] * MB * NB + midx[None, None, None, None, :, None] * NB + nidx[None, None, None, None, None, :]

    tl.store(output_ptr + oidx, ret)


paras_4d = [
    (eval('torch.float32'), 2, 64, 16, 2),
    (eval('torch.float32'), 8, 8, 4, 2),
    (eval('torch.float16'), 2, 64, 16, 2),
    (eval('torch.float16'), 8, 8, 4, 2),
    (eval('torch.int8'), 2, 64, 16, 2),
    (eval('torch.int8'), 8, 8, 4, 2),
]


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('data_type,XB,YB,ZB,MB', paras_4d)
def test_npu_4d(data_type, XB, YB, ZB, MB):
    x = torch.randint(low=-128, high=128, size=(XB, YB, ZB, MB), dtype=data_type).npu()
    expected = x.unsqueeze(2)

    output = torch.randint(1, (XB, YB, 1, ZB, MB), dtype=data_type).npu()

    fn_npu_4d[(1,)](output, x, XB=XB, YB=YB, ZB=ZB, MB=MB)

    torch.testing.assert_close(output, expected)


paras_5d = [
    (eval('torch.float32'), 2, 32, 3, 16, 2),
    (eval('torch.float32'), 8, 8, 3, 4, 2),
    (eval('torch.float16'), 2, 32, 3, 16, 2),
    (eval('torch.float16'), 8, 8, 3, 4, 2),
    (eval('torch.int8'), 2, 32, 3, 16, 2),
    (eval('torch.int8'), 8, 8, 3, 4, 2),
]


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('data_type,XB,YB,ZB,MB,NB', paras_5d)
def test_npu_5d(data_type, XB, YB, ZB, MB, NB):
    x = torch.randint(low=-128, high=128, size=(XB, YB, ZB, MB, NB), dtype=data_type).npu()
    expected = x.unsqueeze(2)

    output = torch.randint(1, (XB, YB, 1, ZB, MB, NB), dtype=data_type).npu()

    fn_npu_5d[(1,)](output, x, XB=XB, YB=YB, ZB=ZB, MB=MB, NB=NB)

    torch.testing.assert_close(output, expected)
