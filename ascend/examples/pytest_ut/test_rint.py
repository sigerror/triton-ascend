# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest

import triton
import triton.language as tl
import time

import torch
import torch_npu
import test_common

@triton.jit
def triton_rint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for loop1 in range(0, XBLOCK, XBLOCK_SUB):
        x0 = offset + loop1 + tl.arange(0,XBLOCK_SUB)
        xmask = x0 < xnumel
        tmp0 = tl.load(in_ptr0 + x0, mask=xmask)
        tmp1 = tl.math.rint(tmp0)
        tl.store(out_ptr0 + x0, tmp1, mask=xmask)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 32, 2048, 64],
                         ])
def test_rint(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch.round(x)
    y_cal = test_common.generate_tensor(x.shape, dtype).npu()
    triton_rint[ncore, 1, 1](x, y_cal, x.numel(), xblock, xblock_sub, debug=True)
    test_common.validate_cmp_with_expection(dtype, y_cal, y_ref, True)

@pytest.mark.parametrize('dtype',['float32',])
def test_rint_half(dtype):
    x0 = torch.tensor([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=eval('torch.' + dtype)).npu()
    y_ref = torch.round(x0)
    y_cal = test_common.generate_tensor(x0.shape, dtype).npu()
    triton_rint[32, 1, 1](x0, y_cal, x0.numel(), 2048, 64, debug=True)
    test_common.validate_cmp_with_expection(dtype, y_cal, y_ref, True)