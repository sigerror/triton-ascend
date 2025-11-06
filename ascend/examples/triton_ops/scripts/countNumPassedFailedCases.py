# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
import re
import sys
import argparse
from pathlib import Path
import logging

DEFAULT_STAT_TUPLE = (0, 0, 0, 0)


def is_test_summary_line(line):
    """ """
    return "failed" in line or "passed" in line or "skipped" in line


def parse_test_summary(line):
    """ """
    # 用于匹配各个测试结果数量的正则表达式
    failed = re.search(r"(\d+)\s+failed", line)
    passed = re.search(r"(\d+)\s+passed", line)
    skipped = re.search(r"(\d+)\s+skipped", line)

    # 提取耗时信息，可以是秒为单位的浮点数，也可以是(0:02:51)这样的格式
    time_match = re.search(r"(\d+\.\d+)\s*s|\((\d+):(\d+):(\d+)\)", line)

    # 获取各个测试结果的数量，若未找到则设为0
    failed_count = int(failed.group(1)) if failed else 0
    passed_count = int(passed.group(1)) if passed else 0
    skipped_count = int(skipped.group(1)) if skipped else 0

    # 计算耗时（秒）
    total_seconds = 0
    if time_match:
        if time_match.group(1):  # 直接匹配到秒数的情况
            total_seconds = float(time_match.group(1))
        else:  # 匹配到(时:分:秒)格式的情况
            hours = int(time_match.group(2))
            minutes = int(time_match.group(3))
            seconds = int(time_match.group(4))
            total_seconds = hours * 3600 + minutes * 60 + seconds

    return failed_count, passed_count, skipped_count, total_seconds


def process_log_file(file_path):
    """

    :param file_path:

    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            if not lines:
                return DEFAULT_STAT_TUPLE
            for line in reversed(lines):
                if is_test_summary_line(line):
                    return parse_test_summary(line)
            return DEFAULT_STAT_TUPLE
    except Exception as e:
        raise OSError(f"Error reading {file_path}: {e}", file=sys.stderr) from e


def process_all_log_files(directory):
    """ """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    total_failed = 0
    total_passed = 0
    total_skipped = 0
    total_time = 0
    processed_files = 0

    for file_path in directory.glob("*.log"):
        if file_path.is_file():
            failed, passed, skipped, time = process_log_file(file_path)
            total_failed += failed
            total_passed += passed
            total_skipped += skipped
            total_time += time
            processed_files += 1

    total_time_in_min = total_time / 60.0
    total = total_failed + total_passed
    pass_rate = total_passed / total * 100
    logger.info(f"Processed {processed_files} log files")
    logger.info(
        f"Total: {total}, Failed: {total_failed}, Passed: {total_passed}, "
        f"PassRate: {pass_rate:.1f}%, TimeCost: {total_time_in_min:.1f}min. "
        f"NOTE: Skipped: {total_skipped}"
    )


def main():
    """ """
    parser = argparse.ArgumentParser(
        description="Count the number of failed and passed cases in a directory"
    )
    parser.add_argument("directory", help="Directory containing the log files")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        raise ValueError(
            f"[Error]: Directory '{directory}' does not exist!", file=sys.stderr
        )

    process_all_log_files(directory)


if __name__ == "__main__":
    main()
