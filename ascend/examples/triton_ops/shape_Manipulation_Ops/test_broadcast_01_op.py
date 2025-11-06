# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import pytest
import triton
import triton.language as tl
import torch
import test_common


@triton.jit
def triton_broadcast_to_dim01(in_ptr0, out_ptr0, L: tl.constexpr,
                              M: tl.constexpr, N: tl.constexpr):
    """
    _summary_

    :param in_ptr0:
    :param out_ptr0:
    :param L:
    :param M:
    :param N:
    """
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = tl.arange(0, 1)[:, None, None] * N * 1 + tl.arange(
        0, 1)[None, :, None] * N + nblk_idx[None, None, :]
    odx = lblk_idx[:, None, None]*N*M + \
        mblk_idx[None, :, None]*N+nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(out_ptr0 + odx)
    ret = tl.broadcast(x, x1)
    tl.store(out_ptr0 + odx, ret)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    shapelist = [
        (37, 5, 3),
    ]
    typelist = [
        "float32",
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
def test_broadcast_01(sigtype, L, M, N):
    """
    _summary_

    :param sigtype:
    :param L:
    :param M:
    :param N:
    """
    dtype = test_common.get_torch_typename(sigtype)
    x0 = test_common.generate_tensor(shape=(1, 1, N), dtype=sigtype).npu()
    ans = x0.repeat(L, M, 1)
    output = torch.zeros((L, M, N), dtype=dtype).npu()
    triton_broadcast_to_dim01[1, 1, 1](x0, output, L, M, N)
    test_common.validate_cmp(sigtype, output, ans)
