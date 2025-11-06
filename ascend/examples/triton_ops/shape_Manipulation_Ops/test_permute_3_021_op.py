# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os
import triton
import triton.language as tl
import torch
import pytest
import test_common


@triton.jit
def triton_permute_3_021(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr,
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
    ret = X.permute(0, 2, 1)
    oidx = xidx[:, None, None] * YB * ZB + zidx[None, :, None] * YB + yidx[
        None, None, :]
    tl.store(output_ptr + oidx, ret)


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


@pytest.mark.parametrize('XB, YB, ZB', shapelist)
@pytest.mark.parametrize(
    "sigtype",
    [
        pytest.param(
            sigtype,
            marks=pytest.mark.skipif(sigtype in ["bfloat16", "bool", "int64"],
                                     reason="Unsupported for now"),
        ) for sigtype in typelist
    ],
)
def test_permute_3_021(sigtype, XB, YB, ZB):
    """
    _summary_

    :param sigtype:
    :param XB:
    :param YB:
    :param ZB:
    """
    dtype = test_common.get_torch_typename(sigtype)
    x0 = test_common.generate_tensor(shape=(XB, YB, ZB), dtype=sigtype).npu()
    ans = x0.permute(0, 2, 1)
    output = torch.zeros((XB, ZB, YB), dtype=dtype).npu()
    triton_permute_3_021[1, 1, 1](output, x0, XB, YB, ZB)
    test_common.validate_cmp(sigtype, output, ans)
