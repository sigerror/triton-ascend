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

import os
import pytest


def pytest_configure(config):
    # 仅在工作节点设置设备ID
    if hasattr(config, "workerinput"):
        worker_id = config.workerinput["workerid"]
        device_id = int(worker_id.replace("gw", ""))
        os.environ["ASCEND_DEVICE_ID"] = str(device_id)
        print(f"\n>> Worker {worker_id} using NPU device {device_id}")


# 可选：设备初始化逻辑
@pytest.fixture(scope="session", autouse=True)
def init_npu_device():
    if "ASCEND_DEVICE_ID" in os.environ:
        device_id = os.environ["ASCEND_DEVICE_ID"]
        # 在此处添加设备初始化代码
        print(f"Initializing NPU device {device_id}")