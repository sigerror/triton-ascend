# triton 总览

## triton op 支持度总览

|                          |        Triton Op       | int8 | int16 | int32 | uint32 | int64 | fp16 | fp32 | bf16 | bool |
|:------------------------:|:----------------------:|------|-------|-------|--------|-------|------|------|------|------|
|       Creation Ops       | arange                 | ✓    | ✓     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | cat                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | full                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | zeros                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | zeros_like             | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | cast                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|  Shape Manipulation Ops  | broadcast              | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | broadcast_to           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | expand_dims            | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | interleave             | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | join                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | permute                | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓    |
|                          | ravel                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | reshape                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | split                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | trans                  | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓    |
|                          | view                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|    Linear Algebra Ops    | dot                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | dot_scaled             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|    Memory/Pointer Ops    | load                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | store                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | make_block_ptr         | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | advance                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|       Indexing Ops       | flip                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | where                  | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓    |
|                          | swizzle2d              | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|         Math Ops         | add                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | sub                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | mul                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | div                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | floordiv(//)           | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | mod                    | ✓    | ✓     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | neg                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | invert(!)              | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | and(&)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | or(\|)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | xor(^)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | not(~)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | lshift(<<)             | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | rshift(>>)             | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | gt                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | ge                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | lt                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | le                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | eq                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | ne                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | logical and            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ✓    |
|                          | logical or             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ✓    |
|                          | abs                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | cdiv                   | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | ceil                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | clamp                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | cos                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | div_rn                 | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | erf                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | exp                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | exp2                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | fdiv                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | floor                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | fma                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | log                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | log2                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | maximum                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | minimum                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | rsqrt                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sigmoid                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sin                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | softmax                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sqrt                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sqrt_rn                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | umulhi                 | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|       Reduction Ops      | argmax                 | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | argmin                 | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | max                    | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | min                    | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | reduce                 | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sum                    | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | xor_sum                | ✓    | ✓     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|       Scan/Sort Ops      | associative_scan       | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | cumprod                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | cumsum                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | histogram              | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | sort                   | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | gather                 | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|        Atomic Ops        | atomic_add             | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_and             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_cas             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_max             | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_min             | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_or              | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_xchg            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_xor             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
| Random Number Generation | randint4x              | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | randint                | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | rand                   | ×    | ×     | ×     | ×      | ×     | ×    | ✓    | ×    | ×    |
|                          | randn                  | ×    | ×     | ×     | ×      | ×     | ×    | ✓    | ×    | ×    |
|         Iterators        | range                  | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | static_range           | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|      Inline Assembly     | inline_asm_elementwise | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|     Compiler Hint Ops    | debug_barrier          | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | max_constancy          | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | max_contiguous         | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | multiple_of            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|         Debug Ops        | static_print           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | static_assert          | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | device_print           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ×    | ✓    |
|                          | device_assert          | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |

## 约束说明

- dot: 两个输入A[batch(optional), M, K], B[batch(optional), K, N]，M，N按照16对齐，K按照32B对齐。

- gather: triton.gather(x, index, axis)，假设x的shape为n维度，目前只支持axis=n-1。

- permute: triton.permute(x, dims)，不支持dims=[2, 1, 0]。

- trans: triton.trans(x, dims)，不支持dims=[2, 1 , 0]。

- device_print: 需要增加2个环境变量，TRITON_DEVICE_PRINT=1，TRITON_ENABLE_TASKQUEUE=0。**TRITON_ENABLE_TASKQUEUE=0可能造成程序运行不稳定，建议仅临时使用。**

- atomic_add: 不支持标量（包括长度为1的tensor）访存

- atomic_max: 不支持标量（包括长度为1的tensor）访存

- atomic_min: 不支持标量（包括长度为1的tensor）访存

- permute: 不支持不相邻轴转置，如`(0, 1, 2) -> (2, 1, 0)`

- trans: 不支持不相邻轴转置，如`(0, 1, 2) -> (2, 1, 0)`

- umulhi: 不支持负数输入

- ALL: int8类型由于特殊处理，会占用更大的片上空间，编译时容易造成ub overflow报错，通常调整tilling即可解决；
       triton kernel中同时存在所有tensor总和不能超过96KB，若关闭double buffer，则不能超过192KB；
       所有tensor不允许某个shape的size小于1。
