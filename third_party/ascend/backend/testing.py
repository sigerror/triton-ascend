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

import builtins
import functools
import logging
import multiprocessing
import threading
import os
import subprocess
import sys
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


import psutil

from triton.runtime import cache


def do_bench_npu(fn, warmup=5, active=30, prof_dir=None, keep_res=False):
    import torch
    import torch_npu

    # warmup kernel
    fn()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    skip_first = 1
    wait = 0
    repeat = 1
    total = skip_first + (wait + warmup + active) * repeat

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(cache.get_home_dir(), ".triton", "profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in builtins.range(total):
            fn()
            prof.step()
            torch.npu.synchronize()

    time = collect_single(torch_path)
    _rm_dic(keep_res, torch_path)
    return time


def collect_single(base_dir: str, key: str = None) -> float:
    if not os.path.exists(base_dir):
        return float("inf")

    import pandas as pd

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != "op_statistic.csv":
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            if key is not None:
                key_rows = df[df["OP Type"].str.startswith(key, na=False)]
                if not key_rows.empty:
                    return key_rows["Avg Time(us)"].values[0]
                return float("inf")
            else:
                # default: read the first row except header
                return df.loc[0, "Avg Time(us)"]

    return float("inf")


def do_bench_multiple_kernel_npu(kernel_dict, active=30, prof_dir=None, keep_res=False):
    import torch
    import torch_npu

    from triton.compiler.errors import (
        CompilationError,
        CompileTimeAssertionFailure,
        MLIRCompilationError,
    )

    assert (
        len(kernel_dict) > 0
    ), f"ERROR: length of kernel_dict is {len(kernel_dict)}, no kernel is profiling."

    kernel_dict_temp_lock = threading.Lock()
    tiling_dict_lock = threading.Lock()
    tiling_dict = {}
    kernel_dict_temp = {}

    # warmup kernel
    def run_fn(config, fn):
        try:
            fn()
            if kernel_dict_temp_lock:
                kernel_dict_temp[config] = fn
        except (CompileTimeAssertionFailure, MLIRCompilationError, CompilationError) as ex:
            if tiling_dict_lock:
                tiling_dict[config] = [float("inf"), float("inf"), float("inf")]

    def run_all_fns():
        max_workers = min(psutil.cpu_count(logical=False) // 2, len(kernel_dict))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for config, fn in kernel_dict.items():
                future = executor.submit(run_fn, config, fn)
                futures.append(future)
            for future in futures:
                try:
                    future.result()
                except Exception as ex:
                    logging.info(f"Exception raised while benchmarking function.{ex}")

    run_all_fns()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(cache.get_home_dir(), ".triton", "profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")

    l2_cache_size = 192 * (1 << 20)
    buffer = torch.empty(l2_cache_size // 4, dtype=torch.int, device="npu")
    buffer.zero_()
    torch.npu.synchronize()  # shake out of any npu error

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _, fn in kernel_dict_temp.items():
            for _ in builtins.range(active):
                buffer.zero_()
                fn()
                torch.npu.synchronize()
    del buffer

    tiling_dict_temp = _collect_mul_prof_result(base_dir=torch_path, kernel_dict=kernel_dict_temp, total=active)
    tiling_dict.update(tiling_dict_temp)
    _rm_dic(keep_res, torch_path)
    return tiling_dict


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _collect_mul_prof_result(base_dir: str, kernel_dict, total, key: str = None):
    import numpy as np
    import pandas as pd

    tiling_dict = {}
    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(kernel_dict)
    if kernel_details_file is None or os.path.exists(kernel_details_file) is False:
        for config, _ in kernel_dict.items():
            tiling_dict[config] = [float("inf")]
        return tiling_dict
    df = pd.read_csv(kernel_details_file)
    # filter out l2 cache clear operation
    filter_cond = ~df["Type"].str.contains(r"^ZerosLike$", case=False, na=False)
    filter_df = df[filter_cond]
    if key is not None:
        key_rows = filter_df[filter_df["Name"].str.contains(key, na=False)]
    else:
        key_rows = filter_df
    time_cost = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, total):
            row_index = active_index + func_idx * total
            time_cost[func_idx] += key_rows.iloc[row_index]["Duration(us)"]
    time_cost = [x / total for x in time_cost]
    for config, avg_time in zip(kernel_dict.keys(), time_cost):
        tiling_dict[config] = [avg_time]
    return tiling_dict
