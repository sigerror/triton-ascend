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

from triton.language import core, math, semantic
from triton._C.libtriton import ir

@core.extern
def reciprocal(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_recipf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_recipDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)

@core.extern
def log1p(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_log1pf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_log1pDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def relu(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_reluf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_reluDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("fp16"),): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("bf16"),): ("__hmf_isinf", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_tanf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_tanDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_atanf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_atanDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)

@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_tanhf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_tanhDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)

@core.extern
def ilogb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_ilogbf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_ilogbDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ldexp(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_ldexpf", core.dtype("fp32")),
            (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_ldexpDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)

@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_powf", core.dtype("fp32")),
            (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_powf", core.dtype("fp16")),
            (core.dtype("bf16"), core.dtype("bf16")): ("__hmf_powf", core.dtype("bf16")),
            (core.dtype("int64"), core.dtype("int64")): ("__hmf_powi", core.dtype("int64")),
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_powi", core.dtype("int32")),
            (core.dtype("int16"), core.dtype("int16")): ("__hmf_powi", core.dtype("int16")),
            (core.dtype("int8"), core.dtype("int8")): ("__hmf_powi", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)

@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("fp16"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("bf16"),): ("__hmf_isnan", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)

@core.extern
def flip(arg0, arg1=None, _builder=None):
    if arg1 == None:
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("bf16"), ): ("__hmf_flipDhb", core.dtype("bf16")),
                (core.dtype("fp16"), ): ("__hmf_flipDh", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_flipf", core.dtype("fp32")),
                (core.dtype("int8"), ): ("__hmf_flipi8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__hmf_flipi16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__hmf_flipi32", core.dtype("int32")),
                (core.dtype("uint32"), ): ("__hmf_flipui32", core.dtype("uint32")),
                (core.dtype("int64"), ): ("__hmf_flipi64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)

    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("bf16"), core.dtype("int32")): ("__hmf_flipDhb", core.dtype("bf16")),
            (core.dtype("fp16"), core.dtype("int32")): ("__hmf_flipDh", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("int32")): ("__hmf_flipf", core.dtype("fp32")),
            (core.dtype("int8"), core.dtype("int32")): ("__hmf_flipi8", core.dtype("int8")),
            (core.dtype("int16"), core.dtype("int32")): ("__hmf_flipi16", core.dtype("int16")),
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_flipi32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("int32")): ("__hmf_flipui32", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int32")): ("__hmf_flipi64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def atan2(arg0, _builder=None):
    core.static_print("tl.atan2 is unsupported for now. Use libdevice.atan2 instead.")
    core.static_assert(False)

@core.extern
def div_rz(arg0, arg1, _builder=None):
    core.static_print("tl.div_rz is unsupported for now. Use libdevice.div_rz instead.")
    core.static_assert(False)

@core.extern
def fmod(arg0, arg1, _builder=None):
    core.static_print("tl.fmod is unsupported for now. Use libdevice.fmod instead.")
    core.static_assert(False)

@core.extern
def trunc(arg0, _builder=None):
    core.static_print("tl.trunc is unsupported for now. Use libdevice.trunc instead.")
    core.static_assert(False)

@core.extern
def round(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_roundf", core.dtype("fp32")),            
        }, is_pure=True, _builder=_builder)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("acosh")
def acosh(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    tmp = semantic.sub(semantic.mul(arg0, arg0, True, _builder), 1.0, True, _builder)
    sqrt_res = core.tensor(_builder.create_sqrt(tmp.handle), tmp.type)
    sum_res = semantic.add(arg0, sqrt_res, True, _builder)
    return core.tensor(_builder.create_log(sum_res.handle), sum_res.type)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("atanh")
def atanh(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    a = semantic.add(1.0, arg0, True, _builder)
    b = semantic.sub(1.0, arg0, True, _builder)
    lna = core.tensor(_builder.create_log(a.handle), a.type)
    lnb = core.tensor(_builder.create_log(b.handle), b.type)
    tmp = semantic.sub(lna, lnb, True, _builder)
    return semantic.mul(tmp, 0.5, True, _builder)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("expm1")
def expm1(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    tmp = core.tensor(_builder.create_exp(arg0.handle), arg0.type)
    return semantic.sub(tmp, 1, True, _builder)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_2arg_docstr("nextafter")
def nextafter(arg0: core.tensor, arg1: core.tensor, _builder: ir.builder):
    x = semantic.to_tensor(arg0, _builder)
    y = semantic.to_tensor(arg1, _builder)
    dtype_map = {
        "bf16": core.int16,
        "fp16": core.int16,
        "fp32": core.int32
    }
    min_pos_bit = {
        "bf16": 0x0001,
        "fp16": 0x0001,
        "fp32": 0x00000001
    }
    max_neg_bit = {
        "bf16": 0x8001,
        "fp16": 0x8001,
        "fp32": 0x80000001
    }
    int_type = dtype_map[x.type.scalar.name]
    x_eq_y = semantic.equal(x, y, _builder)
    x_gt_0 = semantic.greater_than(x, 0, _builder)
    y_gt_x = semantic.greater_than(y, x, _builder)
    next_neg = semantic.xor_(x_gt_0, y_gt_x, _builder)
    next_pos = semantic.not_(next_neg, _builder)

    p1 = semantic.full(x.shape, 1, int_type, _builder)
    n1 = semantic.full(x.shape, -1, int_type, _builder)
    dir_xy = semantic.where(next_pos, p1, n1, _builder)
    x_abs = math.abs(x, _builder=_builder)
    x_is_0 = semantic.equal(x_abs, 0, _builder)

    min_pos = semantic.full(x.shape, min_pos_bit[x.type.scalar.name], int_type, _builder)
    max_neg = semantic.full(x.shape, max_neg_bit[x.type.scalar.name], int_type, _builder)
    min_pos = semantic.bitcast(min_pos, x.dtype, _builder)
    max_neg = semantic.bitcast(max_neg, x.dtype, _builder)
    bits_x = semantic.bitcast(x, int_type, _builder)
    bits_next = semantic.add(bits_x, dir_xy, True, _builder)
    next_val = semantic.bitcast(bits_next, x.dtype, _builder)

    need_min_pos = semantic.logical_and(x_is_0, next_pos, _builder)
    need_max_neg = semantic.logical_and(x_is_0, next_neg, _builder)
    next_val = semantic.where(need_min_pos, min_pos, next_val, _builder)
    next_val = semantic.where(need_max_neg, max_neg, next_val, _builder)
    return semantic.where(x_eq_y, x, next_val, _builder)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_2arg_docstr("hypot(Euclidean Distance)")
def hypot(arg0: core.tensor, arg1: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    arg1 = semantic.to_tensor(arg1, _builder)
    x2 = semantic.mul(arg0, arg0, True, _builder)
    y2 = semantic.mul(arg1, arg1, True, _builder)
    sum_res = semantic.add(x2, y2, True, _builder)
    return core.tensor(_builder.create_sqrt(sum_res.handle), sum_res.type)

# This function is derived from the Cephes Math Library release 2.8: June, 2000
# https://netlib.org/cephes/
# Copyright (c) 1984, 1987, 2000 by Stephen L. Moshier
# All rights reserved.
@core.builtin
@math._check_dtype(dtypes=["fp16", "fp32"])
@math._add_math_2arg_docstr("besseli0 (Modified Bessel function of the first kind, order 0).")
def cyl_bessel_i0(arg0: core.tensor, _builder: ir.builder):
    param1 = [
            -4.41534164647933937950e-18,
            +3.33079451882223809783e-17,
            -2.43127984654795469359e-16,
            +1.71539128555513303061e-15,
            -1.16853328779934516808e-14,
            +7.67618549860493561688e-14,
            -4.85644678311192946090e-13,
            +2.95505266312963983461e-12,
            -1.72682629144155570723e-11,
            +9.67580903537323691224e-11,
            -5.18979560163526290666e-10,
            +2.65982372468238665035e-09,
            -1.30002500998624804212e-08,
            +6.04699502254191894932e-08,
            -2.67079385394061173391e-07,
            +1.11738753912010371815e-06,
            -4.41673835845875056359e-06,
            +1.64484480707288970893e-05,
            -5.75419501008210370398e-05,
            +1.88502885095841655729e-04,
            -5.76375574538582365885e-04,
            +1.63947561694133579842e-03,
            -4.32430999505057594430e-03,
            +1.05464603945949983183e-02,
            -2.37374148058994688156e-02,
            +4.93052842396707084878e-02,
            -9.49010970480476444210e-02,
            +1.71620901522208775349e-01,
            -3.04682672343198398683e-01,
            +6.76795274409476084995e-01,
    ]
    param2 = [
            -7.23318048787475395456e-18,
            -4.83050448594418207126e-18,
            +4.46562142029675999901e-17,
            +3.46122286769746109310e-17,
            -2.82762398051658348494e-16,
            -3.42548561967721913462e-16,
            +1.77256013305652638360e-15,
            +3.81168066935262242075e-15,
            -9.55484669882830764870e-15,
            -4.15056934728722208663e-14,
            +1.54008621752140982691e-14,
            +3.85277838274214270114e-13,
            +7.18012445138366623367e-13,
            -1.79417853150680611778e-12,
            -1.32158118404477131188e-11,
            -3.14991652796324136454e-11,
            +1.18891471078464383424e-11,
            +4.94060238822496958910e-10,
            +3.39623202570838634515e-09,
            +2.26666899049817806459e-08,
            +2.04891858946906374183e-07,
            +2.89137052083475648297e-06,
            +6.88975834691682398426e-05,
            +3.36911647825569408990e-03,
            +8.04490411014108831608e-01,
    ]
    arg0 = semantic.to_tensor(arg0, _builder)
    abs_x = core.tensor(_builder.create_fabs(arg0.handle), arg0.type)
    x_a = semantic.sub(semantic.mul(abs_x, 0.5, True, _builder), 2.0, True, _builder)
    a_n_2 = 0
    a_n_1 = 0
    a_n = param1[0]
    for i in range(1, 30):
        a_n_2 = a_n_1
        a_n_1 = a_n
        a_n = semantic.sub(semantic.mul(x_a, a_n_1, True, _builder), a_n_2, True, _builder)
        a_n = semantic.add(a_n, param1[i], True, _builder)

    f_32 = semantic.full(abs_x.shape, 32.0, abs_x.type.scalar, _builder)
    x_b = semantic.sub(semantic.fdiv(f_32, abs_x, True, _builder), 2.0, True, _builder)
    b_n_2 = 0
    b_n_1 = 0
    b_n = param2[0]
    for i in range(1, 25):
        b_n_2 = b_n_1
        b_n_1 = b_n
        b_n = semantic.sub(semantic.mul(x_b, b_n_1, True, _builder), b_n_2, True, _builder)
        b_n = semantic.add(b_n, param2[i], True, _builder)
    
    half_exp = semantic.mul(core.tensor(_builder.create_exp(abs_x.handle), abs_x.type), 0.5, True, _builder)
    res_a = semantic.mul(half_exp, semantic.sub(a_n, a_n_2, True, _builder), True, _builder)
    res_b = semantic.fdiv(semantic.mul(half_exp, semantic.sub(b_n, b_n_2, True, _builder), True, _builder), \
        core.tensor(_builder.create_sqrt(abs_x.handle), abs_x.type), True, _builder)
    cond = semantic.less_equal(abs_x, 8.0, _builder)
    res = semantic.where(cond, res_a, res_b, _builder)
    return res