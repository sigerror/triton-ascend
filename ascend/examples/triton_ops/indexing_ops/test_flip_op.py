# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice
import torch
import pytest
import test_common


@triton.jit
def triton_flip(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr,
                ZB: tl.constexpr):
    """
    _summary_

    :param output_ptr:
    :param x_ptr:
    :param XB:
    :param YB:
    :param ZB:
    """
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :,
                                               None] * ZB + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    ret = libdevice.flip(X, 2)
    tl.store(output_ptr + idx, ret)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    shapelist = [
        (7, 17, 15),
    ]
    typelist = [
        "float32",
    ]
elif test_group == "full":
    shapelist = [
        (1, 22, 9),
        (27, 1, 9),
        (27, 22, 1),
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
    typelist = [
        'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16',
        'bool'
    ]
else:
    raise ValueError(f"Invalid TRITON_TEST_GROUP {test_group}")


@pytest.mark.parametrize('L, M, N', shapelist)
@pytest.mark.parametrize(
    "sigtype",
    [
        pytest.param(
            sigtype,
            marks=pytest.mark.skipif(sigtype in ["bfloat16", "int64"],
                                     reason="Unsupported for now"),
        ) for sigtype in typelist
    ],
)
def test_flip(sigtype, L, M, N):
    """
    _summary_

    :param sigtype:
    :param L:
    :param M:
    :param N:
    """
    dtype = test_common.get_torch_typename(sigtype)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    ans = torch.flip(x0, dims=(-1, ))
    output = torch.zeros((L, M, N), dtype=dtype).npu()
    triton_flip[1, 1, 1](output, x0, L, M, N)
    test_common.validate_cmp(sigtype, output, ans)
