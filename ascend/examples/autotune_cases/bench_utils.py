import multiprocessing
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Callable, Tuple

import pandas as pd
import torch
import torch_npu


USEC_PER_MSEC = 1e3


class BenchUtils:

    @staticmethod
    def benchmark_test(
        fn_torch: Callable,
        fn_triton: Callable,
        torch_args: Tuple[Any, ...],
        triton_args: Tuple[Any, ...],
        warmup: int = 5,
        repeats: int = 10,
    ) -> Tuple[float, float, float]:
        stream = torch.npu.current_stream()
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)

        # warm_up triton func
        stream.synchronize()
        for _ in range(warmup):
            fn_triton(triton_args)
        stream.synchronize()

        start_event.record()
        for _ in range(repeats):
            fn_triton(triton_args)
        end_event.record()
        stream.synchronize()
        time_triton = start_event.elapsed_time(end_event) / repeats * USEC_PER_MSEC

        # warm_up torch func
        stream.synchronize()
        for _ in range(warmup):
            fn_torch(torch_args)
        stream.synchronize()

        start_event.record()
        for _ in range(repeats):
            fn_torch(torch_args)
        end_event.record()
        stream.synchronize()
        time_eager = start_event.elapsed_time(end_event) / repeats * USEC_PER_MSEC

        perf_ratio = time_eager / time_triton
        return perf_ratio, time_eager, time_triton

    @staticmethod
    def profiling_test(
        fn_torch: Callable,
        fn_triton: Callable,
        torch_args: Tuple[Any, ...],
        triton_args: Tuple[Any, ...],
        warmup: int = 5,
        repeats: int = 30,
    ) -> Tuple[float, float, float]:
        stream = torch.npu.current_stream()

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        )
        skip_first = 1
        wait = 0
        total = skip_first + (wait + warmup + repeats)
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        prof_path = "./profile_result"
        triton_base_path = os.path.join(prof_path, "triton")
        torch_base_path = os.path.join(prof_path, "torch")

        triton_prof_path = os.path.join(
            triton_base_path, f"prof_{timestamp}_{process_name}-{pid}"
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=wait, warmup=warmup, active=repeats, repeat=1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                triton_prof_path
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            stream.synchronize()

            for _ in range(total):
                fn_triton(triton_args)
                prof.step()
            stream.synchronize()

        time_triton = BenchUtils.collect_single(triton_prof_path)

        torch_prof_path = os.path.join(
            torch_base_path, f"prof_{timestamp}_{process_name}-{pid}"
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=wait, warmup=warmup, active=repeats, repeat=1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                torch_prof_path
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            stream.synchronize()

            for _ in range(total):
                fn_torch(torch_args)
                prof.step()
            stream.synchronize()

        time_eager = BenchUtils.collect_single(torch_prof_path)

        if os.path.exists(triton_prof_path):
            shutil.rmtree(triton_prof_path)
        if os.path.exists(torch_prof_path):
            shutil.rmtree(torch_prof_path)

        perf_ratio = time_eager / time_triton
        return perf_ratio, time_eager, time_triton

    @staticmethod
    def collect_single(base_dir: str, key: str = None) -> float:
        if not os.path.exists(base_dir):
            return float("inf")

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

    @staticmethod
    def validate_perf(
        fn_torch: Callable,
        fn_triton: Callable,
        torch_args: Tuple[Any, ...],
        triton_args: Tuple[Any, ...],
        op_name: str,
        ratio: float = 0.8,
        warmup: int = 5,
        repeats: int = 30,
    ) -> None:
        res = BenchUtils.profiling_test(
            fn_torch, fn_triton, torch_args, triton_args, warmup, repeats
        )
        perf_ratio, time_eager, time_triton = res
        if perf_ratio > 0:
            try:
                assert perf_ratio >= ratio
            except AssertionError as e:
                raise AssertionError(
                    f"{op_name}: perf ratio is less than {ratio}: {perf_ratio}, "
                    f"time eager: {time_eager}, time triton: {time_triton}"
                ) from e
        else:
            raise ValueError(
                f"Invalid perf results, [perf_ratio, time_eager, time_triton]: {res}"
            )
