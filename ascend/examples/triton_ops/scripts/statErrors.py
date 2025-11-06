# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
import os
import sys
from collections import defaultdict
import logging


def analyze_logs(directory):
    """

    :param directory:

    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    # Data structure: {error_message: {'count': int, 'files': set}}
    error_dict = defaultdict(lambda: {"count": 0, "files": set()})

    # Validate directory existence
    if not os.path.isdir(directory):
        raise OSError(f"Error: Directory '{directory}' does not exist.")

    # Process each log file in the directory
    for filename in os.listdir(directory):
        if filename.endswith("_summary.log"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as file:
                    for line in file:
                        line = line.strip()
                        if line.startswith("Error:"):
                            error_dict[line]["count"] += 1
                            error_dict[line]["files"].add(filename)
            except Exception as e:
                raise OSError(f"Could not read file {filename}: {str(e)}") from e

    # Output results sorted by error message
    print_info = []
    for error, data in sorted(error_dict.items()):
        print_info.append(f"{error} (Count: {data['count']})\n")
        for file in sorted(data["files"]):
            op = file.split("_summary.log")[0]
            print_info.append(f"  {op},")
        print_info.append("\n")
    print_info.append(f"Number of error types: {len(error_dict)}\n")
    logger.info("\n" + "".join(print_info))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <log_file_dir>")
        sys.exit(1)

    analyze_logs(sys.argv[1])
