# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
import triton
import triton.language as tl
import pytest


BLOCK: tl.constexpr = 32


@triton.jit
def triton_(in_ptr0, out_ptr0, x0_numel, r1_numel, XBLOCK: tl.constexpr,
            XBLOCK_SUB: tl.constexpr, block_id_threshold: tl.constexpr,
            XBLOCK1: tl.constexpr, num_core: tl.constexpr):
    """_summary_

    :param in_ptr0: 
    :param out_ptr0: 
    :param x0_numel: 
    :param r1_numel: 
    :param XBLOCK: 
    :param XBLOCK_SUB: 
    :param block_id_threshold: 
    :param XBLOCK1: 
    :param num_core: 
    """
    RBLOCK: tl.constexpr = 64

    block_idx = tl.program_id(0) * tl.num_programs(1) * tl.num_programs(
        2) + tl.program_id(1) * tl.num_programs(2) + tl.program_id(2)
    if block_idx < block_id_threshold:
        offset = block_idx * XBLOCK
        loops1 = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB  # 32+23 / 24 = 2
        upper = offset + XBLOCK
    else:
        offset = block_id_threshold * XBLOCK + (
            block_idx - block_id_threshold
        ) * XBLOCK1     # pid=34 offset = 9*32 + (34-9)*24 = 888
        loops1 = (XBLOCK1 + XBLOCK_SUB - 1) // XBLOCK_SUB       # 1
        if block_idx == num_core - 1:
            upper = x0_numel
        else:
            upper = offset + XBLOCK1  # 912

    base1 = tl.arange(0, XBLOCK_SUB)
    base2 = tl.arange(0, RBLOCK)

    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1[:, None]
        xmask = x0 < upper
        r1 = base2[None, :]
        rmask = r1 < r1_numel
        tmp0 = tl.load(in_ptr0 + (r1 + (64 * x0)), rmask & xmask, other=0.0)

        tmp1 = tl.reshape(tmp0, [XBLOCK_SUB, RBLOCK])
        tmp2_tmp = tl.sum(tmp1, 1)
        tmp2 = tmp2_tmp.reshape(XBLOCK_SUB, 1)

        tl.store(out_ptr0 + (x0), tmp2, xmask)


guards = {"dummy": None}


# @pytest.mark.skip(reason="multi-process error, to be fixed.")
@pytest.mark.parametrize("size", [(1025, 64)])
def test_program_id(size):
    """_summary_

    :param size: 
    """
    b = torch.randn((size), dtype=torch.float32).npu()
    c = torch.sum(b, dim=1)

    ret = torch.randn((size[0]), dtype=torch.float32).npu()

    triton_[5, 2, 4](b,
                     ret,
                     size[0],
                     size[1],
                     XBLOCK=32,
                     XBLOCK_SUB=16,
                     block_id_threshold=9,
                     XBLOCK1=24,
                     num_core=40,
                     debug=True)
    torch.testing.assert_close(c, ret)


if __name__ == "__main__":
    pytest.main([__file__])
