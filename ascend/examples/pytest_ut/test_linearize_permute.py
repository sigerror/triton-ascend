import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu

types_all = [
    (torch.float32, 'float32'),
]


shapes_common = [
    (8, 2048, 4),
]


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def k_load_perm_select(ptr, out, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = x < xnumel
    ymask = y < ynumel
    bad_mask = xmask | ymask
    val = tl.load(ptr + (x + 4 * y), bad_mask)
    tl.store(out + (x + 4 * y), val, xmask & ymask)


@triton.jit
def k_store_perm_select(in_ptr, out_ptr, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = x < xnumel
    ymask = y < ynumel
    bad_mask = xmask | ymask
    val = tl.load(in_ptr + (x + 4 * y), xmask & ymask)
    tl.store(out_ptr + (x + 4 * y), val, bad_mask)


@triton.jit
def k_load_moddiv_noperm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    # 2D tile：[YBLOCK, XBLOCK]
    y = yindex[:, None]                                  # (YBLOCK, 1)
    x = xindex[None, :]                                  # (1, XBLOCK)

    mask = y < ynumel

    z = (yindex // SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_ = (yindex % SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_linear = z * SHAPE1 + y_                           # (YBLOCK, 1)
    offset_load = x + SHAPE2 * y_linear                  # (YBLOCK, XBLOCK)
    val = tl.load(in_ptr + offset_load, mask=mask)

    offset_store = x + SHAPE2 * y                        # (YBLOCK, XBLOCK)
    tl.store(out_ptr + offset_store, val, mask=mask)


@triton.jit
def k_store_moddiv_noperm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    y = yindex[:, None]                                  # (YBLOCK, 1)
    x = xindex[None, :]                                  # (1, XBLOCK)

    mask = (y < ynumel) & (x < xnumel)

    offset_load = x + SHAPE2 * y                         # (YBLOCK, XBLOCK)
    val = tl.load(in_ptr + offset_load, mask=mask)

    z = (yindex // SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_ = (yindex % SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_linear = z * SHAPE1 + y_                           # (YBLOCK, 1)
    offset_store = x + SHAPE2 * y_linear                 # (YBLOCK, XBLOCK)
    tl.store(out_ptr + offset_store, val, mask=mask)


@triton.jit
def k_load_moddiv_perm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    X = xindex[:, None]                                 # (XBLOCK, 1)
    Y = yindex[None, :]                                 # (1, YBLOCK)
    mask_load = (X < xnumel) & (Y < ynumel)             # (XBLOCK, YBLOCK)

    z = (yindex // SHAPE1)[None, :]                     # (1, YBLOCK)
    y_ = (yindex % SHAPE1)[None, :]                     # (1, YBLOCK)
    offset_load = X + SHAPE2 * y_ + SHAPE2 * SHAPE1 * z # (XBLOCK, YBLOCK)
    val = tl.load(in_ptr + offset_load, mask=mask_load) # (XBLOCK, YBLOCK)

    y2 = yindex[:, None]                                # (YBLOCK, 1)
    x2 = xindex[None, :]                                # (1, XBLOCK)
    mask_store = (y2 < ynumel) & (x2 < xnumel)          # (YBLOCK, XBLOCK)
    offset_store = x2 + SHAPE2 * y2                     # (YBLOCK, XBLOCK)

    tl.store(out_ptr + offset_store, val.permute(1, 0), mask=mask_store)


@triton.jit
def k_store_moddiv_perm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    y = yindex[:, None]                                  # (YBLOCK, 1)
    x = xindex[None, :]                                  # (1, XBLOCK)
    mask_load = (y < ynumel) & (x < xnumel)

    offset_load = x + SHAPE2 * y                         # (YBLOCK, XBLOCK)
    val = tl.load(in_ptr + offset_load, mask=mask_load)  # (YBLOCK, XBLOCK)

    X = xindex[:, None]                                  # (XBLOCK, 1)
    Y = yindex[None, :]                                  # (1, YBLOCK)
    mask_store = (X < xnumel) & (Y < ynumel)             # (XBLOCK, YBLOCK)

    z = Y // SHAPE1                                      # (1, YBLOCK)
    y_ = Y % SHAPE1                                      # (1, YBLOCK)
    offset_store = X + SHAPE2 * y_ + SHAPE2 * SHAPE1 * z # (XBLOCK, YBLOCK)

    tl.store(out_ptr + offset_store, val.permute(1, 0), mask=mask_store)


@triton.jit
def k_load_store_moddiv_noperm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    y = yindex[:, None]                                  # (YBLOCK, 1)
    x = xindex[None, :]                                  # (1, XBLOCK)

    mask = (y < ynumel) & (x < xnumel)

    z = (yindex // SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_ = (yindex % SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_linear = z * SHAPE1 + y_                           # (YBLOCK, 1)

    offset = x + SHAPE2 * y_linear                       # (YBLOCK, XBLOCK)

    val = tl.load(in_ptr + offset, mask=mask)
    val = val + 2
    tl.store(out_ptr + offset, val, mask=mask)


@triton.jit
def k_load_store_moddiv_perm(
    in_ptr, out_ptr, 
    ynumel, xnumel, 
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    # Program ID
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    # Block index calculation
    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # (XBLOCK,)

    # Broadcasting to 2D
    x = xindex[:, None]  # (XBLOCK, 1)
    y = yindex[None, :]  # (1, YBLOCK)

    # Valid mask
    mask = (x < xnumel) & (y < ynumel)

    # Simulate linear index back to 3D
    z = y // SHAPE1  # (1, YBLOCK)
    y_ = y % SHAPE1  # (1, YBLOCK)

    # compute input offset (simulate [z, y_, x] access in strided format)
    offset = x + SHAPE2 * y_ + SHAPE2 * SHAPE1 * z  # (XBLOCK, YBLOCK)

    # load with implicit transpose (will be interpreted as [Y, X] then transposed)
    val = tl.load(in_ptr + offset, mask=mask)

    # apply some dummy operation
    val = val + 1

    # store it back to out_ptr with same offset
    tl.store(out_ptr + offset, val, mask=mask)


@pytest.mark.parametrize('dtype,sigtype', types_all)
@pytest.mark.parametrize('xnumel, ynumel, XBLOCK, YBLOCK', [(3, 513, 4, 64)])
def test_k_load_perm_select(xnumel, ynumel, XBLOCK, YBLOCK, dtype, sigtype):
    # # Fix for int8 not supported
    # if sigtype == 'int8':
    #     pytest.skip(f"int8 not supported")

    in_ptr = test_common.generate_tensor(shape=(ynumel * 4,), dtype=sigtype).npu()
    out_ptr = torch.zeros_like(in_ptr)


    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    k_load_perm_select[grid](in_ptr, out_ptr, ynumel, xnumel, YBLOCK=YBLOCK, XBLOCK=XBLOCK)

    out_ref = torch.zeros_like(out_ptr)
    y_idx = torch.arange(ynumel).unsqueeze(1).npu()        # [ynumel, 1]
    x_idx = torch.arange(xnumel).unsqueeze(0).npu()        # [1, xnumel]
    idx = (x_idx + 4 * y_idx).reshape(-1)                          # [ynumel * xnumel]
    out_ref[idx] = in_ptr[idx]
    torch.testing.assert_close(out_ptr[idx], out_ref[idx])


@pytest.mark.parametrize('dtype,sigtype', types_all)
@pytest.mark.parametrize('xnumel, ynumel, XBLOCK, YBLOCK', [(3, 513, 4, 64)])
def test_k_store_perm_select(xnumel, ynumel, XBLOCK, YBLOCK, dtype, sigtype):
    in_ptr = test_common.generate_tensor(shape=(ynumel * 4,), dtype=sigtype).npu()
    out_ptr = torch.zeros_like(in_ptr)

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    k_store_perm_select[grid](in_ptr, out_ptr, ynumel, xnumel, YBLOCK=YBLOCK, XBLOCK=XBLOCK)

    out_ref = torch.zeros_like(out_ptr)
    y_idx = torch.arange(ynumel).unsqueeze(1).npu()
    x_idx = torch.arange(xnumel).unsqueeze(0).npu()
    idx = (x_idx + 4 * y_idx).reshape(-1)
    out_ref[idx] = in_ptr[idx]
    torch.testing.assert_close(out_ptr[idx], out_ref[idx])


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_load_moddiv_noperm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.zeros_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64
    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_load_moddiv_noperm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X
    )

    torch.testing.assert_close(out, in_flat)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_store_moddiv_noperm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.zeros_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64
    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_store_moddiv_noperm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X
    )

    torch.testing.assert_close(out, in_flat)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_load_moddiv_perm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.zeros_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_load_moddiv_perm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X
    )

    torch.testing.assert_close(out, in_flat)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_store_moddiv_perm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.zeros_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_store_moddiv_perm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X
    )

    torch.testing.assert_close(out, in_flat)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_load_store_moddiv_noperm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.empty_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = 4
    YBLOCK = 256
    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_load_store_moddiv_noperm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X
    )

    ref = (a + 2).contiguous().view(-1)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_load_store_moddiv_perm(Z, Y, X, dtype, sigtype):
    shape = (Z, Y, X)
    numel = Z * Y * X

    a = test_common.generate_tensor(shape, dtype=sigtype).npu()
    a_flat = a.contiguous().view(-1)

    out = torch.zeros_like(a_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = 4
    YBLOCK = 256
    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_load_store_moddiv_perm[grid](
        a_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X
    )

    a_reshaped = a + 1
    out_ref = a_reshaped.contiguous().view(-1)

    torch.testing.assert_close(out, out_ref)
