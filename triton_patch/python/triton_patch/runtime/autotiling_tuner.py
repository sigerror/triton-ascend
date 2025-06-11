from __future__ import annotations

import builtins
import os
import time
from typing import Dict, List

from .autotuner import Autotuner, Config


class AutoTilingTuner(Autotuner):

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
        split_params=None,
        tiling_params=None,
        low_dims=None,
        dual_reduction=False,
        persistent_reduction=False
    ):
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
            do_bench
        )

        if not configs:
            self.user_configs = []
        else:
            self.user_configs = configs
        self.gen_configs = []   # generated configs from TileGenerator

        self.split_params = split_params
        self.tiling_params = tiling_params
        self.low_dims = low_dims
        self.dual_reduction = dual_reduction
        self.persistent_reduction = persistent_reduction

    def _gen_tile_configs(self, kv_dict: Dict[str, int], dtype) -> List[Config]:
        from .tile_generator import KernelMeta, TileGenerator

        _dim_names = {'x', 'y', 'z', 'w', 'v', 't', 'rx', 'ry', 'rz', 'rw', 'rv', 'rt'}
        axis_sizes = {}
        for k, v in kv_dict.items():
            if k not in _dim_names:
                continue
            if not isinstance(v, int):
                raise ValueError(f"Not supported dim type: {type(v)}, `int` is the only supported type")
            
            axis_sizes[k] = v

        # check if split & tiling axis's name in axis names
        for k in (list(self.split_params.keys()) + list(self.tiling_params.keys())):
            if k not in axis_sizes and ('r' + k) not in axis_sizes:
                raise KeyError(f"Cannot identify {k} axis's size")
            
        # check if low_dims's name in axis names
        for k in self.low_dims:
            if k not in axis_sizes and ('r' + k) not in axis_sizes:
                raise KeyError(f"Unknown low dims name: {k}, should be in {axis_sizes.keys()}")
            
        kernel_meta = KernelMeta(axis_sizes, self.split_params, self.tiling_params, self.low_dims,
                                    dtype, self.persistent_reduction, self.dual_reduction)
        tile_gen = TileGenerator(kernel_meta=kernel_meta)
        tile_gen.descend_split_tiling()

        self.gen_configs.clear()
        self.gen_configs = list(tile_gen.configs.values())

        if len(self.gen_configs) == 0 and len(self.user_configs) == 0:
            return [
                Config({}, num_warps=4, num_stages=2, num_ctas=1, num_buffers_warp_spec=0, num_consumer_groups=0,
                       reg_dec_producer=0, reg_inc_consumer=0)
            ]
        else:
            return self.gen_configs + self.user_configs

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True

        # generate key
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
        _kv_dict = {k: _args[v] for k, v in self.keys.items() if v in _args}
        key = list(_kv_dict.values())
        # Currently, we assume the input & output's dtype is the same
        dtype = None
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
                dtype = arg.dtype
        if dtype is None:
            raise ValueError("Cannot identify the inputs or outputs' data type!")
        
        key = tuple(key)
        if key not in self.cache:
            # prune configs
            self.configs = self._gen_tile_configs(_kv_dict, dtype)
            pruned_configs = self.prune_configs(kwargs)
            if len(pruned_configs) > 1:
                used_cached_result = False
                bench_start = time.time()
                timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
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
            print(f"Triton autotuning for function {self.base_fn.__name__} finished after "
                  f"{self.bench_time:.2f}s; best config selected: {self.best_config};")
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