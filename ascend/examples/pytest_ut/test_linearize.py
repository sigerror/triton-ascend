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

import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu
from triton.runtime.driver import driver



# npu hardware params from trion
target = driver.active.get_current_target()
device = driver.active.get_current_device()
prop = driver.active.utils.get_device_properties(device)

num_cube_core = prop["num_aicore"]
num_vector_core = prop["num_aicore"]
if (target.arch in ("Ascend910B","Ascend910_9382" )):
    num_vector_core = num_cube_core * 2
    print(target.arch, "vector_core",num_vector_core)

def foo(a, d ,shape ):
    y = a.reshape(shape)
    y = y.permute(0,2,1) + d
    return y


@triton.jit
def triton_gpu_revised(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, 
                       SHAPE0:tl.constexpr, SHAPE1:tl.constexpr,SHAPE2:tl.constexpr,
                       XBLOCK : tl.constexpr):
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x2 = xindex[:, None]
    x2_1 = xindex[None, :]
    y3 = yindex[ :, None]
    y0 = (yindex % SHAPE1)[None,:]
    y1 = (yindex // SHAPE1)[None,:]
    tmp0 = tl.load(in_ptr0 + (x2_1 + (SHAPE2*y3)), (x2_1 < xnumel) & (y3 < ynumel) )
    tmp1 = tl.load(in_ptr1 + (y0 + (SHAPE1*x2) + (SHAPE1*SHAPE2*y1)), \
                   (xindex[:,None] < xnumel) & (yindex[None,:] < ynumel))
                   # (x2 < xnumel) & (y0 < SHAPE1))
                   # (xindex[:,None] < xnumel) & (yindex[None,:] < ynumel))
    tmp10 = tmp1.permute(1,0)
    tmp2 = tmp0 + tmp10
    tl.store(out_ptr0 + (x2_1 + (SHAPE2*y3)), tmp2, (x2_1 < xnumel) & (y3 < ynumel))

def biggest_divisor(num):
    for i in range(2,num):  
        if num % i == 0:  
            return num // i
    return num



def find_good_yblock(ynumel,xnumel, y_upper, dtype) :
    y = ynumel
    x = xnumel

    align_numel = 4 if dtype==torch.int64 else 8 
    ub_upper = 3900 if dtype==torch.int64 else 8000

 
    # optimize block_dim
    def get_block_dim(y,x) :
        return ((xnumel + x -1 )//x) * ((y_upper+ y -1 ) // y)
    
    count = 0
    while( get_block_dim(y,x) < num_vector_core and y > 8 and count < 20 ) :
        y_1 = biggest_divisor(y) 
        if get_block_dim(y_1, x) > num_vector_core :
            break
        y = y_1
        if  get_block_dim(y,x) < num_vector_core and x > align_numel :
            x = x // 2 
        count = count + 1

    # optimize block_size to avoid ub-overflow
    while( y * x > ub_upper) :
        y_1 = biggest_divisor(y)
        if y_1 == y or y_1 <=align_numel:
            break
        y = y_1

    while( y * x > ub_upper and x > align_numel) :
        x_1 = x // 2
        if x_1 <= align_numel :
            break
        x = x_1
            
    return (x,y)

def triton_foo(a, d ,shape, dtype) :
    z, y, x = shape
    out = torch.empty_strided((z, x, y), (x*y, 1, x), device='npu', dtype=dtype)
    XBLOCK, YBLOCK= find_good_yblock(y,x, y*z, dtype=dtype)
    print(f"XBLOCK={XBLOCK},YBLOCK={YBLOCK}, block_dim={((x + XBLOCK -1 )//XBLOCK) * (((y*z) + YBLOCK -1 ) // YBLOCK)}")
    grid = ((x + XBLOCK -1 )//XBLOCK, ((y*z) + YBLOCK -1 ) // YBLOCK, 1) 
    
    triton_gpu_revised[grid](a, d, out, y*z, x, 
                             SHAPE0=z,SHAPE1=y,SHAPE2=x,
                             YBLOCK=YBLOCK, XBLOCK = XBLOCK)
    return out 


types = [
    (torch.float32, 'float32'),
]


shapes = [(8,2048,4)]

@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('Z,Y,X', shapes)
def test_linearize(Z,Y,X, dtype, sigtype) :
    shape = (Z,Y,X)
    print(f"start data validation on shape:{shape}")
    a = test_common.generate_tensor(shape=(Z, Y * X), dtype=sigtype).npu()
    d = test_common.generate_tensor(shape=(Z, X, Y), dtype=sigtype).npu()
    r = triton_foo( a,  d, shape, dtype)
    r1 = foo(a, d ,shape )
    test_common.validate_cmp(sigtype, r1, r)
    print(f"data validation passed")


# Test linearize offset handling with expert routing pattern
@triton.jit
def linearize_offset_kernel(
    bias_ptr,
    output_ptr,
    experts_ids_ptr,
    N: tl.constexpr,
    EM: tl.constexpr,
    stride_bias_e,
    stride_bias_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    off_experts = tl.load(experts_ids_ptr + pid_m).to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
        tl.store(output_ptr + pid*16 + tl.arange(0, 16), bias.reshape(16))


def torch_linearize_offset(bias_ptr, experts_ids_ptr, N, EM, stride_bias_e, stride_bias_n, 
                           BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):
    """PyTorch reference implementation for offset handling"""
    output = torch.empty([16, 16], dtype=bias_ptr.dtype, device=bias_ptr.device)
    
    num_pid_m = (EM + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    for pid in range(16):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        off_experts = experts_ids_ptr[pid_m].to(torch.int64).item()
        offs_bn = torch.arange(pid_n * BLOCK_SIZE_N, 
                               (pid_n + 1) * BLOCK_SIZE_N, 
                               dtype=torch.int64, device=bias_ptr.device) % N
        
        bias = bias_ptr[off_experts, offs_bn]
        output[pid] = bias
    
    return output


@pytest.mark.parametrize('dtype,sigtype', [(torch.float32, 'float32')])
def test_linearize_offset_handling(dtype, sigtype):
    """
    Test linearization's handling of complex offset patterns.
    This test simulates expert routing scenarios where offsets are computed
    dynamically based on expert IDs, validating correct pointer arithmetic
    and linearization in the compiler.
    """
    print(f"Testing linearize offset handling with dtype={sigtype}")
    
    # Setup test data
    num_experts = 4
    hidden_dim = 64
    bias_ptr = torch.arange(0, num_experts * hidden_dim, dtype=dtype).npu().reshape(num_experts, hidden_dim)
    output_ptr = torch.empty([16, 16], dtype=dtype).npu()
    experts_ids_ptr = torch.tensor([1, 2, 3, 1], dtype=torch.int32).npu()
    
    # Kernel parameters
    N = 64
    EM = 64
    stride_bias_e = 64
    stride_bias_n = 1
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    GROUP_SIZE_M = 4
    
    # Run triton kernel
    linearize_offset_kernel[(16,)](
        bias_ptr=bias_ptr,
        output_ptr=output_ptr,
        experts_ids_ptr=experts_ids_ptr,
        N=N,
        EM=EM,
        stride_bias_e=stride_bias_e,
        stride_bias_n=stride_bias_n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    # Compute reference result
    expected = torch_linearize_offset(
        bias_ptr, experts_ids_ptr, N, EM, 
        stride_bias_e, stride_bias_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )
    
    # Validate results
    test_common.validate_cmp(sigtype, expected, output_ptr)
    print(f"Linearize offset handling test passed")