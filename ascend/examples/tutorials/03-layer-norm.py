"""
Layer Normalization
=============
"""
import pytest
import torch
import triton
import triton.language as tl
import torch_npu

@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=[],
    key={'x': "M", 'y': "N"},
    split_params={'x': "XBLOCK_SIZE"},
    tiling_params={'y': "RBLOCK_SIZE"},
    low_dims=['y'],
    persistent_reduction=False,
    dual_reduction=False
)
@triton.jit
def _layer_norm_fwd_fused_autotune(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,
    M,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    XBLOCK_SIZE: tl.constexpr,
    RBLOCK_SIZE: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row_begin = tl.program_id(0) * XBLOCK_SIZE
    row_idx = row_begin + tl.arange(0, XBLOCK_SIZE)
    row_mask = row_idx < M
    row_offsets = row_idx[:,None]*stride
    # Compute mean

    _mean = tl.zeros((XBLOCK_SIZE, RBLOCK_SIZE), dtype=tl.float32)
    for off in range(0, N, RBLOCK_SIZE):
        col_idx = off + tl.arange(0, RBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:,None] & col_mask[None,:]
        a = tl.load(X + row_offsets + col_idx[None,:], mask=mask, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1, keep_dims = True) / N

    # Compute variance
    _var = tl.zeros((XBLOCK_SIZE, RBLOCK_SIZE), dtype=tl.float32)
    for off in range(0, N, RBLOCK_SIZE):
        col_idx = off + tl.arange(0, RBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:,None] & col_mask[None,:]
        x = tl.load(X + row_offsets + col_idx[None,:], mask=mask, other=0.).to(tl.float32)
        x = tl.where(mask, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=1, keep_dims=True) / N

    rstd = 1 / tl.sqrt(var + eps)
    
    # Write mean / rstd
    tl.store(Mean + row_idx[:,None], mean, mask = row_mask[:,None])
    tl.store(Rstd + row_idx[:,None], rstd, mask = row_mask[:,None])

    # Normalize and apply linear transformation
    for off in range(0, N, RBLOCK_SIZE):
        col_idx = off + tl.arange(0, RBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:,None] & col_mask[None,:]
        w = tl.load(W + col_idx, mask=col_mask).reshape((1, RBLOCK_SIZE))
        b = tl.load(B + col_idx, mask=col_mask).reshape((1, RBLOCK_SIZE))
        x = tl.load(X + row_offsets + col_idx[None,:], mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + row_offsets + col_idx[None,:], y, mask=mask)


@torch.inference_mode()
def layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    kernel = _layer_norm_fwd_fused[(M, )](  #
        x_arg, y, weight, bias, mean, rstd,  #
        x_arg.stride(0), N, eps,  #
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
    # print(kernel.asm['ttir'])
    return y


@torch.inference_mode()
def layer_norm_autotune(x, normalized_shape, weight, bias, eps=1e-5):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta["XBLOCK_SIZE"]), 1, 1)
    # enqueue kernel
    _layer_norm_fwd_fused_autotune[grid](  #
        x_arg, y, weight, bias, mean, rstd,  #
        x_arg.stride(0), N, M, eps)
    # print(kernel.asm['ttir'])
    return y


def _layer_norm(M, N, dtype, eps=1e-5, device='npu'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_tri_autotune = layer_norm_autotune(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(y_tri_autotune, y_ref, atol=1e-2, rtol=0)
    print(f"y_tri: {y_tri}")
    print(f"y_tri_autotune: {y_tri_autotune}")
    print(f"y_ref: {y_ref}")
    print(f"Layer Normalization {M},{N} {dtype} PASSED!")

if __name__ == '__main__':
    _layer_norm(128, 128, torch.float16)
    _layer_norm(128, 128, torch.bfloat16)
    _layer_norm(128, 128, torch.float32)
