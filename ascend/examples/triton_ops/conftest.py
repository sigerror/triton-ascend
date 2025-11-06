# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
import logging
import pytest


@pytest.fixture(autouse=True)
def setup_logging():
    """ """
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
