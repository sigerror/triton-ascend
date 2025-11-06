# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def vector_kernel(x_ptr, y):
    """_summary_

    :param x_ptr: 
    :param y: 
    """
    x_ptrs = x_ptr + tl.arange(0, 8)
    x = tl.load(x_ptrs)
    tl.debug_barrier()
    tl.device_print("x", x)
    tl.device_print("y and 16", y, 16, hex=True)


def test_vector():
    """_summary_
    """
    x = torch.ones((8, ), device="npu", dtype=torch.float32)
    vector_kernel[(2, )](x, 15)


DEVICE = "npu"


def get_ascend_autotune_config():
    """_summary_

    :return: 
    """
    return [
        triton.Config({
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 16,
            'BLOCK_SIZE_K': 16,
            # 'GROUP_SIZE_M': 1
        }),
    ]


@triton.autotune(
    configs=get_ascend_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        # M,
        # N,
        # K,
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
):
    """_summary_

    :param a_ptr: 
    :param b_ptr: 
    :param c_ptr: 
    :param stride_am: 
    :param stride_ak: 
    :param stride_bn: 
    :param stride_cn: 
    :param BLOCK_SIZE_M: 
    :param BLOCK_SIZE_N: 
    :param BLOCK_SIZE_K: 
    """
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_n[None, :] * stride_bn)
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a = tl.load(a_ptrs)
    tl.device_print("a1 = ", a)
    b = tl.load(b_ptrs)
    tl.device_print("b2 = ", b)
    tl.device_print("c3 = ", c)
    c = tl.dot(a, b, c)
    c16 = c.to(tl.float16)
    tl.device_print("c4 = ", c)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c16)


def matmul(a, b):
    """_summary_

    :param a: 
    :param b: 
    :return: 
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    K = K
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    matmul_kernel[(1, )](
        a,
        b,
        c,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        debug=True)
    return c


def test_device_print():
    """_summary_
    """
    a = torch.ones((16, 16), device=DEVICE, dtype=torch.float32)
    v = 0
    for i in range(16):
        for j in range(16):
            a[i][j] = v
            v += 1
    b = torch.ones((16, 16), device=DEVICE, dtype=torch.float32)
    return matmul(a, b)


if __name__ == "__main__":
    try:
        test_device_print()
    except Exception:
        sys.exit(1)
    sys.exit(0)
