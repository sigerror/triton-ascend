# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import pytest
import triton
import triton.language as tl
import torch
import test_common


@triton.jit
def triton_atomic_min(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr,
                      N: tl.constexpr):
    """_summary_

    :param in_ptr0: 
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
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[
        None, :, None] * N + nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    tl.atomic_min(out_ptr0 + odx, x)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    shapelist = [
        (2, 16, 5),
    ]
    typelist = [
        "int32",
    ]
elif test_group == "full":
    shapelist = [
        (1, 22, 39),
        (27, 1, 39),
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
        (23, 5, 31),
        (7, 11, 32),
        (7, 11, 33),
        (2, 3, 255),
        (3, 3, 256),
        (3, 2, 257),
    ]

    typelist = ['int8', 'int16', 'int32', 'float16', 'float32', 'bfloat16']
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
def test_atomic_min(sigtype, L, M, N):
    """_summary_

    :param sigtype: 
    :param L: 
    :param M: 
    :param N: 
    """
    x0 = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    output = test_common.generate_tensor(shape=(L, M, N), dtype=sigtype).npu()
    ans = torch.minimum(x0, output)
    triton_atomic_min[1, 1, 1](x0, output, L, M, N, debug=True)
    test_common.validate_cmp(sigtype, output, ans)
