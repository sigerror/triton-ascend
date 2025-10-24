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
