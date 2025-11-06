# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
import functools
import re
import torch
import pytest

_float_dtypes = ["float32", "float16", "bfloat16"]
_int_dtypes = ["int32", "int64", "int16", "int8"]
_int_dtypes_without_int64 = ["int32", "int16", "int8"]
_all_dtypes_no_bool = _float_dtypes + _int_dtypes
_all_dtypes = _all_dtypes_no_bool + ["bool"]
_32bit_dtypes = ["float32", "int32"]
_16bit_dtypes = ["float16", "bfloat16", "int16"]
_float_dtypes_without_bf16 = ["float32", "float16"]

_shape_1d = [1, 3, 17, 32, 741]
_shape_5d = [
    (2, 2, 2, 2, 8),
    (3, 1, 3, 5, 7),
    (3, 7, 5, 3, 1),
]


def generate_tensor(shape, dtype):
    """

    :param shape:
    :param dtype:

    """
    if dtype in ["float32", "float16", "bfloat16"]:
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ["int32", "int64", "int16"]:
        return torch.randint(
            low=-2000, high=2000, size=shape, dtype=eval("torch." + dtype)
        )
    if dtype == "int8":
        return torch.randint(
            low=-128, high=127, size=shape, dtype=eval("torch." + dtype)
        )
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()

    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def fill_zero_with_one(x):
    """

    :param x:

    """
    return x.masked_fill(x == 0, 1)


def fill_negative_with_one(x):
    """

    :param x:

    """
    return x.masked_fill(x < 0, 1)


def get_triton_sig_typename(dtype):
    """

    :param dtype:

    """
    if dtype == "float32":
        tyname = "*fp32"
    elif dtype == "int32":
        tyname = "*i32"
    elif dtype == "int64":
        tyname = "*i64"
    elif dtype == "float16":
        tyname = "*fp16"
    elif dtype == "int16":
        tyname = "*i16"
    elif dtype == "int8":
        tyname = "*i8"
    elif dtype == "bool":
        tyname = "*i1"
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
    return tyname


def get_torch_typename(sigtype: str) -> torch.dtype:
    """

    :param sigtype: str:

    """
    type_mapping = {
        "bool": torch.bool,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if sigtype not in type_mapping:
        supported_types = ", ".join(type_mapping.keys())
        raise ValueError(
            f"Unsupported dtype: '{sigtype}'. Supported dtype: {supported_types}"
        )

    return type_mapping[sigtype]


# Relative error: abs(x_ref - x_cal) / abs(x_ref)
# Absolute error: abs(x_ref - x_cal)


# calculation type operators require different error range
# It is a stricter verification and not satisfied now, save it here
def validate_cal(dtype, y_cal, y_ref):
    """

    :param dtype:
    :param y_cal:
    :param y_ref:

    """
    if dtype == "float16":
        if torch.mean(y_ref) < 0.001:
            assert (
                torch.abs(y_cal - y_ref) < 0.001
            ), "|y_cal - y_ref| < 0.001 is required !"
        else:
            diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.001
            # all true
            assert diff.all(), "Relative error is less than 0.001 !"
    if dtype == "float32":
        if torch.mean(y_ref) < 0.0001:
            assert (
                torch.abs(y_cal - y_ref) < 0.0001
            ), "|y_cal - y_ref| < 0.0001 is required !"
        else:
            diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.0001
            assert diff.all(), "Relative error is less than 0.001 !"
    elif dtype == "bfloat16":
        diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.001
        assert diff.all(), "Relative error is less than 0.001 !"
    elif dtype in ["int32", "int64", "int16"]:
        assert torch.equal(y_cal, y_ref)
    elif dtype == "bool":
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


# moving and comparison ops require no precision error
def validate_cmp(dtype, y_cal, y_ref):
    """

    :param dtype:
    :param y_cal:
    :param y_ref:

    """
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()
    if dtype == "float16":
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == "bfloat16":
        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cal.to(torch.float32),
            rtol=1e-03,
            atol=1e-03,
            equal_nan=True,
        )
    elif dtype == "float32":
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype in ["int32", "int64", "int16", "int8"]:
        assert torch.equal(y_cal, y_ref)
    elif dtype == "bool":
        # WRT bool tensor, we need to copy to host to compare.
        # With tensor on device, y_cal == y_ref returns list containing False
        # even if they looks like all equal. No idea why.
        assert torch.equal(y_cal.cpu(), y_ref.cpu())
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def validate_cmp_with_expection(dtype, y_cal, y_ref, expect):
    """

    :param dtype:
    :param y_cal:
    :param y_ref:
    :param expect:

    """
    if dtype in ["float32", "float16", "bfloat16"]:
        if expect:
            assert torch.allclose(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
        else:
            assert not torch.allclose(
                y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True
            )
    elif dtype in ["int32", "int64", "int16", "int8"]:
        if expect:
            assert torch.equal(y_cal, y_ref)
        else:
            assert not torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def raises_with_match(expected_exception, match_pattern):
    """

    :param expected_exception:
    :param match_pattern:

    """

    def decorator(test_func):
        """

        :param test_func:

        """

        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            """

            :param *args:
            :param **kwargs:

            """
            with pytest.raises(expected_exception, match=match_pattern):
                return test_func(*args, **kwargs)

        return wrapper

    return decorator


def capture_output(expected_output):
    """

    :param expected_output:

    """

    def decorator(test_func):
        """

        :param test_func:

        """

        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            """

            :param *args:
            :param **kwargs:

            """
            capsys = kwargs.pop("capsys", None)
            if capsys is None:
                try:
                    capsys = pytest.fixture(capsys)()
                except Exception as e:
                    raise RuntimeError(
                        "This decorator requires pytest's capsys fixture"
                    ) from e
            test_func(capsys, *args, **kwargs)
            captured = capsys.readouterr()
            # pybind11::scoped_ostream_redirect captures std::cout with \x00 inserted
            # for now, no idea how to eliminate \x00 from C++ side.
            cleaned = re.sub(r"\x00", "", captured.out)
            assert expected_output in cleaned

        return wrapper

    return decorator
