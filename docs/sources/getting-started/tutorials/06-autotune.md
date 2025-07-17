# 自动调优 （Autotune）

在本节中，我们将展示使用 Triton 的 autotune 方法以自动选择最优的 kernel 配置参数。

说明：
当前Triton-Ascend autotune支持block size、multibuffer（编译器的优化），未来还会持续增加autotune可调项。与社区相比，Triton-Ascend支持bishengir在MLIR编译过程中失败后继续测试下一组config，而非中断autotune。

```Python
import torch, torch_npu
import triton
import triton.language as tl

def test_triton_autotune():

    # 返回一组不同的 kernel 配置，用于 autotune 测试
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': False}),
            triton.Config({'XS': 8 * 1024, 'multibuffer': True}),
        ]
    # 使用 @autotune 装饰器自动选择最优的 kernel 配置，autotune里可以接收的参数意义
    # :param key: axis name: argument name组成的字典，argument 变化会触发候选配置的重新生成与评估. 
    #     axis name 属于集合 {'x','y','z','w','v','t','rx','ry','rz','rw','rv','rt},前缀 'r' 表示规约轴.
    #     只有此参数中的轴名称在作为规约轴时才应该添加前缀 r.
    # :type key: Dict[str, str]
    # :param split_params:axis name: argument name组成的字典, the argument 是切分轴的可调参数, 例如 'XBLOCK'.
    #     The axis name必须在参数key的轴名称集合里. 请勿在轴名称前添加前缀 r. 此参数可以为空， 当split_params 和 tiling_params 都为空的时候不会进行自动寻优。
    #     切分轴通常可以根据 `tl.program_id()` 分核语句来确定。
    # :type split_params: Dict[str, str]
    # :param tiling_params: axis name: argument name组成的字典， the argument is 是分块轴的可调参数, such as 'XBLOCK_SUB'.
    #     axis name必须在参数key的轴名称集合里. 请勿在轴名称前添加前缀 r.
    #     这个参数可以设置为空. 当split_params 和 tiling_params 都为空的时候不会进行自动寻优。
    #     分块轴通常可以根据 `tl.arange()` 分块表达式来确定.
    # :type tiling_params: Dict[str, str]
    # :param low_dims: 所有低维轴的轴名称列表，axis name必须在参数key的轴名称集合里， 请勿在轴名称前添加前缀 r.
    # :type low_dims: List[str]
    # :param dual_reduction: 在多个轴上做规约，影响tiling策略。
    # :param persistent_reduction: 在规约轴上是否做切分，影响tiling策略
    # 使用可以参考 ascend\examples\autotune_cases 里面的案例
    
    @triton.autotune(
        configs=get_autotune_config(),      # 配置列表
        key=['XS', 'multibuffer'],          # 选择输入变量作为选择依据
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        XS: tl.constexpr                  # 块大小，用于控制每个线程块处理多少数据
    ):
        pid = tl.program_id(0)            # 获取当前 program 的 ID
        idx = pid * XS + tl.arange(0, XS) # 当前线程块处理的 index 范围
        msk = idx < numel                 # 避免越界的掩码

        # 重复执行一些计算以模拟负载（并测试性能）/ Repeat computation to simulate load (for perf test)
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)  # 加载 x0
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)  # 加载 x1
            tmp2 = tl.math.exp(tmp0) + tmp1 + i                # 计算
            tl.store(out_ptr0 + idx, tmp2, mask=msk)           # 存储到输出

    # Triton 调用函数，自动使用 autotuned kernel
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["XS"]), 1, 1)  # 计算 grid 大小 
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # 使用 PyTorch 作为参考实现进行对比
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"                         # 使用 NPU 作为设备
    DTYPE = torch.float32
    N = 192 * 1024                      # 输入长度
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)  # 随机输入 x0
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)  # 随机输入 x1
    torch_ref = torch_calc_func(x0, x1)              # 得到参考结果
    triton_cal = triton_calc_func(x0, x1)            # 运行 Triton kernel
    torch.testing.assert_close(triton_cal, torch_ref)  # 验证输出是否一致

if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")  # 输出成功标志 / Print success message
```
