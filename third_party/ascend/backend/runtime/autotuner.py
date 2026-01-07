# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

from __future__ import annotations

import builtins
import logging
import os
import time
from typing import Dict, List

from triton.runtime.autotuner import Autotuner, Config

from .utils import get_byte_per_numel, is_valid_axis_name, valid_axis_names
from .autoparser import SplitAxesParser, TilingAxesParser, ReductionAxesParser, LowDimsAxesParser, PtrNumsParser


class AutoTilingTuner(Autotuner):
    """
    Automatic generateing candidate tiling configs and evaluating their performance to get the best config.
    """

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        auto_profile_dir=None
    ):
        """
        :param key: a list of argument name, where the change of arguments in value will triger re-generating candidates configs and evaluating.
            The parameters in the list will be assigned axis names in sequence, with the axis name being in
            {'x','y','z','w','v','t','rx','ry','rz','rw','rv','rt}, where the prefix 'r' means a reduction axis.
            Only the axis name in this param should add perfix 'r' if it's a reduction axis.
        :type key: List[str]
        """
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )
        self.auto_profile_dir = auto_profile_dir

        if not configs:
            self.user_configs = []
        else:
            self.user_configs = configs
        self.gen_configs = []  # generated configs from TileGenerator

        self.split_params = None
        self.tiling_params = None
        self.low_dim_axes = None
        self.reduction_axes = None
        self.dual_reduction = False
        self.persistent_reduction = False
        self.input_ptr_num = -1
        if len(key) > len(valid_axis_names):
            raise ValueError("Number of parameters exceeds the number of available axes.")
        self.keys = {axis: param for axis, param in zip(valid_axis_names, key)}
        self.print_autotuning = os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1"

    def _gen_tile_configs(
        self, kv_dict: Dict[str, int], dtype: torch.dtype
    ) -> List[Config]:
        from .tile_generator import KernelMeta, TileGenerator

        axis_sizes = {}
        for k, v in kv_dict.items():
            if not is_valid_axis_name(k):
                continue
            if not isinstance(v, int):
                raise ValueError(
                    f"Not supported dim type: {type(v)}, `int` is the only supported type"
                )
            axis_sizes[k] = v

        kernel_meta = KernelMeta(
            axis_sizes,
            self.split_params,
            self.tiling_params,
            self.low_dim_axes,
            dtype,
            self.persistent_reduction,
            self.dual_reduction,
            self.input_ptr_num,
        )
        tile_gen = TileGenerator(kernel_meta=kernel_meta)
        tile_gen.descend_split_tiling()

        self.gen_configs.clear()
        self.gen_configs = tile_gen.configs
        if len(self.gen_configs) == 0:
            print(
                "[WARNING] The generated candidate tiling configs are empty based on provided parameters!"
            )

        if len(self.gen_configs) == 0 and len(self.user_configs) == 0:
            return [
                Config(
                    {},
                    num_warps=4,
                    num_stages=2,
                    num_ctas=1,
                    num_buffers_warp_spec=0,
                    num_consumer_groups=0,
                    reg_dec_producer=0,
                    reg_inc_consumer=0,
                )
            ]
        else:
            return self.gen_configs + self.user_configs

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True

        # generate key
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
        key = [_args[v] for _, v in self.keys.items() if v in _args]

        # Currently, we use the dtype with maximum byte length
        dtype = None
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
                dtype = (
                    arg.dtype
                    if get_byte_per_numel(arg.dtype) >= get_byte_per_numel(dtype)
                    else dtype
                )
        if dtype is None:
            raise NotImplementedError("Not support for non-Tensor inputs")

        key = tuple(key)
        if key not in self.cache:
            miss_params = [arg for arg in self.arg_names if arg not in all_args.keys()]
            # parse pointer params nums
            if self.input_ptr_num == -1:
                self.input_ptr_num = self.autoparse_ptr_nums(miss_params)
            
            # parse autotiling axes
            # reduction axis must be parsed before other axes. it will alter the key
            if not self.reduction_axes:
                self.reduction_axes = self.autoparse_reduction_axes()
            if len(self.reduction_axes) >= 2:
                self.dual_reduction = True

            if not self.low_dim_axes:
                self.low_dim_axes = self.autoparse_low_dim_axes()

            if len(self.reduction_axes) == 1 and \
               self.reduction_axes[0] == self.low_dim_axes[0] and \
               all_args.get(self.keys[self.reduction_axes[0]]) < 1024:
                self.persistent_reduction = True

            if not self.split_params:
                self.split_params = self.autoparse_split_params(miss_params)
            miss_params = [arg for arg in miss_params if arg not in self.split_params.values()]
            if not self.tiling_params:
                self.tiling_params = self.autoparse_tiling_params(miss_params)
            miss_params = [arg for arg in miss_params if arg not in self.tiling_params.values()]
            if miss_params:
                raise ValueError(
                    f"Missing required arguments: {miss_params}. "
                    f"These arguments must be explicitly provided and cannot be automatically tuned. "
                    f"Please ensure that these arguments are passed when calling the function."
                )

            # prune configs
            _kv_dict = {k: _args[v] for k, v in self.keys.items() if v in _args}
            self.configs = self._gen_tile_configs(_kv_dict, dtype)
            pruned_configs = self.prune_configs(kwargs)
            if len(pruned_configs) > 1:
                used_cached_result = False
                bench_start = time.time()
                timings = self._batch_bench(*args, configs=pruned_configs, **kwargs)
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
                config = self.cache[key]
            else:
                config = pruned_configs[0]
        else:
            config = self.cache[key]

        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )

        if not used_cached_result and self.auto_profile_dir is not None:
            self._profile(*args, config=self.best_config, **kwargs)
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def _batch_bench(self, *args, configs, **kwargs):
        kernel_dict = {config: self._tiling_kernel(*args, config=config, **kwargs) for config in configs}
        tiling_dict = self._batch_benchmark(kernel_dict=kernel_dict, quantiles=(0.5, 0.2, 0.8))
        if self.print_autotuning:
            print(f"triton configs: {tiling_dict}")
        return tiling_dict

    def _tiling_kernel(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(full_nargs)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(full_nargs, exception=None)
        return kernel_call

    def _batch_benchmark(self, kernel_dict, rep=10, quantiles=None):
        """
            Benchmark the runtime of the provided function.
            By default, return the median runtime of :code:`fn` along with
            the 20-th and 80-th performance percentile.

            :param kernel_dict: Function to benchmark
            :type kernel_dict: Callable
            :param rep: Repetition time (in ms)
            :type rep: int
            :param quantiles: Performance percentile to return in addition to the median.
            :type quantiles: list[float], optional
        """
        assert len(kernel_dict) > 0, f"ERROR: length of kernel_dict is empty."
        from triton.compiler.errors import CompileTimeAssertionFailure, MLIRCompilationError, CompilationError
        from triton.runtime.errors import OutOfResources
        if self.do_bench.__module__ == "triton.testing":
            enable_bench_npu = os.getenv("TRITON_BENCH_METHOD", 'default').lower() == 'npu'
            if enable_bench_npu:
                from ..testing import do_bench_multiple_kernel_npu
                tiling_dict = do_bench_multiple_kernel_npu(kernel_dict, active=max(30, rep), prof_dir=None, keep_res=False)
                return tiling_dict
        tiling_dict = {}
        for config, kernel_call in kernel_dict.items():
            try:
                tiling_dict[config] = self.do_bench(kernel_call, quantiles=quantiles)
            except (OutOfResources, CompileTimeAssertionFailure, MLIRCompilationError) as ex:
                tiling_dict[config] = [float("inf"), float("inf"), float("inf")]
        return tiling_dict

    def _profile(self, *args, config, **meta):
        from ..testing import do_bench_npu

        kernel_call = self._tiling_kernel(*args, config=config, **meta)
        do_bench_npu(
            kernel_call, prof_dir=self.auto_profile_dir, keep_res=True
        )

    def autoparse_split_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the split axis parameters from triton kernel code.
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting split params parsing...")
        func_ast = self.fn.parse()
        parser = SplitAxesParser(func_ast, self.keys, candidates_params)
        split_axes = parser.parse()
        if self.print_autotuning:
            print(
                f"Triton autotuning: Split params parsing complete. "
                f"Split params: {split_axes}"
            )
        return split_axes

    def autoparse_tiling_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the tiling axis parameters from triton kernel code.
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting tiling params parsing...")
        func_ast = self.fn.parse()
        parser = TilingAxesParser(func_ast, self.keys, candidates_params)
        tiling_axes = parser.parse()
        if self.print_autotuning:
            print(
                f"Triton autotuning: Tiling params parsing complete. "
                f"Tiling params: {tiling_axes}"
            )
        return tiling_axes
    
    def autoparse_reduction_axes(self) -> List[str]:
        """
        Extracts the reduction axis parameters from triton kernel code.
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting reduction params parsing...")
        func_ast = self.fn.parse()
        parser = ReductionAxesParser(func_ast, self.keys)
        reduction_axes = parser.parse()
        for axis in reduction_axes:
            self.keys[f"r{axis}"] = self.keys.pop(axis)
        reduction_axes = [f"r{axis}" for axis in reduction_axes]

        if self.print_autotuning:
            print(
                f"Triton autotuning: Reduction params parsing complete. "
                f"Reduction params: {reduction_axes}"
            )
        return reduction_axes

    def autoparse_low_dim_axes(self) -> List[str]:
        """
        Extracts the low dimension axis from triton kernel code.
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting Low dims axes parsing...")
        func_ast = self.fn.parse()
        parser = LowDimsAxesParser(func_ast, self.keys)
        low_dim_axes = parser.parse()
        if self.print_autotuning:
            print(
                f"Triton autotuning: Low dims axes parsing complete. "
                f"Keys: {self.keys}, Low dims: {low_dim_axes}"
            )
        return low_dim_axes
    
    def autoparse_ptr_nums(self, miss_params: List[str]) -> int:
        """
        Counts the number of pointer parameters from triton kernel code.
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting ptr nums parsing...")
        func_ast = self.fn.parse()
        parser = PtrNumsParser(func_ast, self.keys, miss_params)
        ptr_nums, ptr_params = parser.parse()
        if self.print_autotuning:
            print(
                f"Triton autotuning: Pointer nums parsing complete. "
                f"Pointer params: {ptr_params}, pointer nums: {ptr_nums}"
            )
        return ptr_nums


def ascend_autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
                    warmup=None, rep=None, use_cuda_graph=False, do_bench=None, auto_prof_dir=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    """

    def decorator(fn):
        return AutoTilingTuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                               post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                               use_cuda_graph=use_cuda_graph, do_bench=do_bench, auto_profile_dir=auto_prof_dir)

    return decorator
