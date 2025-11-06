# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import triton
import triton.language as tl
import test_common


import torch
import pytest



@triton.jit
def fn_npu(output_ptr, x_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, dtype: tl.constexpr):
    """

    :param output_ptr:
    :param x_ptr:
    :param X: tl.constexpr:
    :param Y: tl.constexpr:
    :param Z: tl.constexpr:

    """
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    idx = xidx[:, None, None]*Y*Z+yidx[None, :, None]*Z+zidx[None, None, :]
    input_value = tl.load(x_ptr+idx)
    ret = tl.cast(input_value, dtype=dtype)
    tl.store(output_ptr+idx, ret)


to_sigtypelist = ['bool',
                  'int8',
                  'int16',
                  'int32',
                  'int64',
                  'float32',
                  'float16',
                  'bfloat16']
sigtypelist = ['bool',
               'int8',
               'int16',
               'int32',
               'int64',
               'float32',
               'float16',
               'bfloat16']
shapelist = [
    (1, 1, 1),
    (1, 1, 23),
    (23, 1, 1),
    (1, 23, 1),
    (37, 5, 3),
    (2, 29, 4),
    (7, 31, 7),
    (3, 5, 8),
    (7, 17, 15),
    (25, 5, 16),
    (13, 5, 31),
    (9, 11, 32),
    (7, 11, 33),
]


@pytest.mark.parametrize(
    'to_sigtype',
    [
        pytest.param(
            to_sigtype,
            marks=pytest.mark.skipif(
                to_sigtype in ["int64", "bfloat16"], reason="Unsupported for now"
            ),
        )
        for to_sigtype in to_sigtypelist
    ],
)
@pytest.mark.parametrize(
    "sigtype",
    [
        pytest.param(
            sigtype,
            marks=pytest.mark.skipif(
                sigtype in ["int64", "bfloat16"], reason="Unsupported for now"
            ),
        )
        for sigtype in sigtypelist
    ],
)
@pytest.mark.parametrize('X, Y, Z', shapelist)
def test_cast(to_sigtype, sigtype, X, Y, Z):
    """

    :param to_sigtype:
    :param sigtype:
    :param L:
    :param M:
    :param N:

    """
    to_dtype = test_common.get_torch_typename(to_sigtype)
    x = test_common.generate_tensor((X, Y, Z), sigtype).npu()
    a = x.cpu().to(to_dtype)
    output = torch.randint(1, (X, Y, Z), dtype=to_dtype).npu()
    tl_dtype = tl.int1 if to_sigtype == 'bool' else eval('tl.' + to_sigtype)
    fn_npu[1, 1, 1](output, x, X, Y, Z, tl_dtype)
    test_common.validate_cmp(to_sigtype, output, a)
