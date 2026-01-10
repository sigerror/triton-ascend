# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

from typing import (
    TypeVar, List
)

from triton._C.libtriton import ir
import triton.language.core as tl

from . import core as bl


T = TypeVar('T')


def alloc(
    etype: tl.dtype,
    shape: List[tl.constexpr],
    address_space: bl.address_space,
    builder: ir.builder
) -> bl.buffer:
    shape = tl._unwrap_shape(shape)
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be list/tuple")
    etype = tl._constexpr_to_value(etype)
    element_ty = etype.to_ir(builder)
    address_space = tl._constexpr_to_value(address_space)
    addr_space_attr = address_space.to_ir(builder) if address_space else builder.get_null_attr()
    return bl.buffer(builder.allocate_local_buffer(element_ty, shape, addr_space_attr),
                     dtype=etype, shape=shape, space=address_space)


def to_tensor(
    memref: bl.buffer,
    writable: bool,
    builder: ir.builder
) -> tl.tensor:
    if not isinstance(memref, bl.buffer):
        raise TypeError("memref must be bl.buffer")
    tensor_type = tl.block_type(memref.dtype, memref.shape)
    return tl.tensor(builder.to_tensor(memref.handle, writable), tensor_type)
