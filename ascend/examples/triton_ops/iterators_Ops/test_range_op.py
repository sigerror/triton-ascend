# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import pytest
import triton
import triton.language as tl
import torch
import test_common


@triton.jit
def triton_range(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr,
                 N: tl.constexpr):
    """_summary_

    :param in_ptr0: 
    :param in_ptr1: 
    :param out_ptr0: 
    :param L: 
    :param M: 
    :param N: 
    """
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[
        None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0 + x1
    for i in tl.range(2, 5, 2):
        i = i
        ret = ret + x1

    for i in tl.static_range(2, 10, 3):
        i = i
        ret = ret + x0

    odx = lblk_idx[:, None, None] * N * M + mblk_idx[
        None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    shapelist = [
        (3, 5, 8),
    ]
    typelist = [
        "int8",
    ]

elif test_group == "full":
    shapelist = [
        (3, 5, 8),
    ]
    typelist = [
        'int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32'
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
def test_range(sigtype, L, M, N):
    """_summary_

    :param sigtype: 
    :param L: 
    :param M: 
    :param N: 
    """
    dtype = test_common.get_torch_typename(sigtype)
    shape = (L, M, N)
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    y_ref = x0 + x1 + x1 + x1 + x0 + x0 + x0
    output = torch.zeros(shape, dtype=dtype).npu()
    triton_range[1, 1, 1](x0, x1, output, L, M, N)
    test_common.validate_cmp(sigtype, output, y_ref)
