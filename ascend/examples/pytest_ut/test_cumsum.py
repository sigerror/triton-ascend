import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime.libentry import libentry
import pytest
from test_common import _all_dtypes_no_bool, generate_tensor, validate_cmp


def torch_func(x, dim, reverse):
    if reverse:
        x = torch.flip(x, [dim])
    res = torch.cumsum(x, dim=dim)
    return res


@libentry()
@triton.jit
def triton_kernel(
    out_ptr0,
    in_ptr0,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    numel_x: tl.constexpr,
    numel_r: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def triton_func(x, dim, reverse):
    res = torch.empty_like(x)
    triton_kernel[1, 1, 1](
        res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[0], x.shape[1]
    )
    return res

@pytest.mark.skip(reason="waiting for bishengir-compile to support")
@pytest.mark.parametrize("dtype", _all_dtypes_no_bool)
@pytest.mark.parametrize("shape", [(7, 23)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsum(dtype, shape, dim, reverse):
    torch.npu.set_device(3)
    x0 = generate_tensor(shape=shape, dtype=dtype)
    triton_cal = triton_func(x0, dim, reverse)
    torch_ref = torch_func(x0, dim, reverse)
    validate_cmp(dtype, torch_ref, triton_cal)
