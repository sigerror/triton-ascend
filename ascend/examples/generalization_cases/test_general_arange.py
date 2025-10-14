# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math
import pytest
import torch
import triton
import triton.language as tl

import test_common
from test_common import TestUtils


def torch_pointwise(length):
    res = (torch.arange(0, length) / 2.7) * torch.arange(0, length)
    return res


@triton.jit
def triton_arange(out_ptr0, length: tl.constexpr, numel: tl.constexpr):
    offs = tl.program_id(0) * length
    idx = offs + tl.arange(0, length)
    a = idx / 2.7
    b = idx * a
    mask = idx < numel
    tl.store(out_ptr0 + idx, b, mask)


@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', ['int32', 'int16', 'int8', 'int64'])
def test_case(dtype, shape):
    x0 = test_common.generate_tensor(shape, dtype).npu()

    numel = x0.numel()
    ncore = 32 if dtype == 'int8' and numel > 127 else 1
    if dtype in ('float16', 'bfloat16', 'float32', 'bool'):
        # tl.arange doesn't support float and bool
        xblock = numel / ncore
    else:
        xblock = math.ceil(numel / ncore)

    y_ref = torch_pointwise(numel)
    y_cal = torch.zeros(shape, dtype=torch.float32).npu()

    triton_arange[ncore, 1, 1](y_cal, xblock, numel)

    test_common.validate_cmp(dtype, y_cal, y_ref)
