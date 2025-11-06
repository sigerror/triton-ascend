# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
import json
import sys
import logging


def main():
    '''
    '''
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    if len(sys.argv) != 3:
        logger.info(f"Usage: {sys.argv[0]} <json filename> [730|830]")
        sys.exit(1)

    json_path = sys.argv[1]
    version = sys.argv[2]

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.info(f"[ERROR]: {json_path} is not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.info(f"[ERROR]: {json_path} is invalid!")
        sys.exit(1)

    if version not in data:
        logger.info(f"[ERROR]: target ops group {version} does not exist!")
        sys.exit(1)

    for key, value in data[version].items():
        print(f"{key},{value}")


if __name__ == "__main__":
    main()
