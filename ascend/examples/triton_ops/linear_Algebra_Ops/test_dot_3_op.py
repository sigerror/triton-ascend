# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import triton
import triton.language as tl
import torch
import pytest
import test_common


def torch_dot_None(x0, x1):
    """_summary_

    :param x0: 
    :param x1: 
    :return: 
    """
    res = torch.matmul(x0, x1)
    return res


@triton.jit
def triton_dot_3(output_ptr, x_ptr, y_ptr, z_ptr, A: tl.constexpr,
                 B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    """_summary_

    :param output_ptr: 
    :param x_ptr: 
    :param y_ptr: 
    :param z_ptr: 
    :param A: 
    :param B: 
    :param C: 
    :param D: 
    """
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    Xidx = aidx[:, None, None] * B * C + bidx[None, :,
                                              None] * C + cidx[None, None, :]
    Yidx = aidx[:, None, None] * C * D + cidx[None, :,
                                              None] * D + didx[None, None, :]
    Zidx = aidx[:, None, None] * B * D + bidx[None, :,
                                              None] * D + didx[None, None, :]
    X = tl.load(x_ptr + Xidx)
    Y = tl.load(y_ptr + Yidx)
    Z = tl.load(z_ptr + Zidx)
    ret = tl.dot(X, Y, Z)
    oidx = aidx[:, None, None] * B * D + bidx[None, :,
                                              None] * D + didx[None, None, :]
    tl.store(output_ptr + oidx, ret)


test_group = os.getenv("TRITON_TEST_GROUP", "simple").lower()
if test_group == "simple":
    shapelist = [
        (3, 16, 32, 16),
    ]
    typelist = [
        "int8",
    ]
elif test_group == "full":
    shapelist = [
        (3, 16, 32, 16),
        (2, 32, 64, 16),
    ]

    typelist = ['int8', 'float16', 'float32', 'bfloat16']
else:
    raise ValueError(f"Invalid TRITON_TEST_GROUP {test_group}")


@pytest.mark.parametrize('A, B, C, D', shapelist)
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
def test_dot_3(sigtype, A, B, C, D):
    """_summary_

    :param sigtype: 
    :param A: 
    :param B: 
    :param C: 
    :param D: 
    """
    torch.manual_seed(30)
    x0 = test_common.generate_tensor(shape=(A, B, C), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(A, C, D), dtype=sigtype).npu()
    if 'int' in sigtype:
        x2 = test_common.generate_tensor(shape=(A, B, D), dtype='int32').npu()
        ans = torch_dot_None(x0.to(torch.float32), x1.to(torch.float32)).to(
            torch.int32) + x2
        output = torch.zeros((A, B, D), dtype=torch.int32).npu()
        triton_dot_3[1, 1, 1](output, x0, x1, x2, A, B, C, D, debug=True)
        test_common.validate_cmp('int32', output, ans)
    else:
        x2 = test_common.generate_tensor(shape=(A, B, D),
                                         dtype='float32').npu()
        if sigtype == 'bfloat16':
            ans = torch_dot_None(x0.to(torch.float32), x1.to(
                torch.float32)).to(torch.float32) + x2
        else:
            ans = torch_dot_None(x0, x1).to(torch.float32) + x2
        output = torch.zeros((A, B, D), dtype=torch.float32).npu()
        triton_dot_3[1, 1, 1](output, x0, x1, x2, A, B, C, D, debug=True)
        test_common.validate_cmp(sigtype, output, ans)
