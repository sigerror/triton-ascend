# 矩阵乘法 （Matrix Multiplication）

在本节中，我们展示了使用 Triton 进行矩阵乘法的内核实现。

## 计算内核

```Python
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
@triton.jit
def triton_dot_2_Bias(output_ptr, x_ptr, y_ptr, z_ptr,A : tl.constexpr,B : tl.constexpr,C : tl.constexpr):
    bidx=tl.arange(0,A)
    cidx=tl.arange(0,B)
    didx=tl.arange(0,C)
    Xidx=bidx[:,None]*B+cidx[None,:]
    Yidx=cidx[:,None]*C+didx[None,:]
    Zidx=bidx[:,None]*C+didx[None,:]
    X = tl.load(x_ptr+Xidx)
    Y = tl.load(y_ptr+Yidx)
    Z = tl.load(z_ptr+Zidx)
    ret = tl.dot(X, Y) + Z
    oidx=bidx[:,None]*C+didx[None,:]
    tl.store(output_ptr+oidx,ret)
```

## 工具方法

```Python
def torch_dot_Bias(x0, x1, bias):
    res = torch.matmul(x0, x1) + bias
    return res

def get_torch_typename(dtype):
    if dtype == 'float32':
        tyname = torch.float32
    elif dtype == 'int32':
        tyname = torch.int32
    elif dtype == 'int64':
        tyname = torch.int64
    elif dtype == 'float16':
        tyname = torch.float16
    elif dtype == 'int16':
        tyname = torch.int16
    elif dtype == 'int8':
        tyname = torch.int8
    elif dtype == 'bool':
        tyname = torch.bool
    elif dtype == 'bfloat16':
        tyname = torch.bfloat16
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname

def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def validate_cmp(dtype, y_cal, y_ref):
    y_cal=y_cal.npu()
    y_ref=y_ref.npu()
    if dtype == 'float16': 
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32),  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
```

## 参数化测试

```Python
testlist = [   
    (16, 16, 16), 
]

typelist = ['float16',]

@pytest.mark.parametrize('A, B, C',testlist)
@pytest.mark.parametrize('sigtype',typelist)
def test_dot_2_Bias(sigtype, A, B, C):
    dtype = get_torch_typename(sigtype)
    x0 = generate_tensor(shape = (A, B),dtype = sigtype).npu()
    x1 = generate_tensor(shape = (B, C),dtype = sigtype).npu()
    if 'int' in sigtype:
        bias = generate_tensor(shape = (A, C),dtype = 'int32').npu()
        ans = torch_dot_Bias(x0.to(torch.float32), x1.to(torch.float32), bias.to(torch.float32)).to(dtype)
    else:
        bias = generate_tensor(shape = (A, C),dtype = 'float32').npu()
        ans = torch_dot_Bias(x0, x1, bias).to(eval(f"torch.{dtype}"))
    output = torch.zeros((A, C), dtype = dtype).npu()
    triton_dot_2_Bias[1,1,1](output, x0, x1, bias, A, B, C, debug = True)
    validate_cmp(sigtype,output,ans)
    print(f"Test matmul with dtype={sigtype}, shape=({A},{B},{C}) PASSED!")

if __name__ == "__main__":
    test_dot_2_Bias("float16", 16, 16, 16)
```

Out:

```Python
Test matmul with dtype=float16, shape=(16,16,16) PASSED!
```

上面输出日志表明Triton和Pytorch上的输出结果完全一致。
