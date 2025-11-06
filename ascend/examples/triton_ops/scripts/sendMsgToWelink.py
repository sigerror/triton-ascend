# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
import logging
import re
import sys
import os

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)


def send_msg(content, receiver, auth, sender=None):
    """ """
    url = "http://xiaoluban.rnd.huawei.com:80/"
    data = {"content": content, "receiver": receiver, "auth": auth}
    if sender:
        data["sender"] = sender
    res = requests.post(url=url, json=data)
    if not res.ok:
        logger.info(res.text)


def extract_data_from_log(file_path):
    """从日志文件中提取统计数据"""
    try:
        with open(file_path, "r") as file:
            content = file.read()
            #
            pattern = (
                r"Total: (\d+), Failed: (\d+), Passed: (\d+), PassRate: (\d+\.\d+)"
            )
            match = re.search(pattern, content)
            data = {}
            if match:
                data.update(
                    {
                        "Total": int(match.group(1)),
                        "Failed": int(match.group(2)),
                        "Passed": int(match.group(3)),
                        "PassRate": float(match.group(4)),
                    }
                )
            #
            pattern = r"is saved to (/[\w/-]+/?)$"
            match = re.search(pattern, content[-1024:], re.MULTILINE)
            if match:
                data["CacheURL"] = match.group(1)
            return data
    except FileNotFoundError:
        logger.info(f"[ERROR]: '{file_path}' does not exist!")
        return None
    except Exception as e:
        logger.info(f"[ERROR]: Unknown error: {e}")
        return None


def extract_error_info(file_path):
    """从日志文件中提取错误信息"""
    error_pattern = re.compile(r"^Error: (.*?)\(Count: (\d+)\)\s*$", re.MULTILINE)
    op_pattern = re.compile(r"^\s+([a-zA-Z0-9_,\s]+),\s*$", re.MULTILINE)
    total_errors_pattern = re.compile(r"^Number of error types: (\d+)$", re.MULTILINE)

    error_list = []

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 提取错误总数
    total_errors_match = total_errors_pattern.search(content)
    total_errors = int(total_errors_match.group(1)) if total_errors_match else 0

    # 提取每种错误类型及其操作
    error_matches = error_pattern.finditer(content)
    for error_match in error_matches:
        error_type = error_match.group(1).strip()
        count = int(error_match.group(2))

        # 查找当前错误类型后的操作列表
        start_pos = error_match.end()
        op_match = op_pattern.search(content, start_pos)
        ops = []
        if op_match:
            ops_str = op_match.group(1)
            ops = [op.strip() for op in ops_str.split(",") if op.strip()]

        error_info = {"error_type": error_type, "count": count, "operations": ops}
        error_list.append(error_info)

    return {"total_errors": total_errors, "error_list": error_list}


def format_error_info(error_info):
    """格式化错误信息以便输出"""
    formatted = []
    for error in error_info["error_list"]:
        formatted.append(f"Error: {error['error_type']}") # (Count: {error['count']})
        op_line = "  " + ",  ".join(error["operations"]) + ","
        formatted.append(op_line)
    formatted.append(f"Number of error types: {error_info['total_errors']}")
    return "\n".join(formatted)


def get_msg(log_fpath, branch, commit_id):
    """生成测试日志"""
    data = extract_data_from_log(log_fpath)
    error_info_dict = extract_error_info(log_fpath)
    error_info = format_error_info(error_info_dict)
    _, log_fname = os.path.split(log_fpath)
    log_fname_without_ext, _ = os.path.splitext(log_fname)
    # log_fname_without_ext is TARGET_OPS_GROUP_NAME
    job_name = f"Triton Daily Tests({log_fname_without_ext})"
    total = data["Total"]
    passed = data["Passed"]
    failed = data["Failed"]
    pass_rate = data["PassRate"]
    cache_url = data["CacheURL"]
    server_ip = "10.50.90.108"
    msg = (
        f"<span style='color:blue;font-weight:bold;'>---------------{job_name} 任务执行完成---------------</span>\n"
        f"<span style='color:black;font-weight:bold;'>Total: {total}, </span>"
        f"<span style='color:green;font-weight:bold;'>Passed: {passed}, PassRate: {pass_rate}%, </span>"
        f"<span style='color:red;font-weight:bold;'>Failed: {failed}</span>\n"
        f"---------------------------------------------------\n"
        f"<span style='color:black;font-weight:bold;'>日志文件及编译中间产物: {server_ip}:{cache_url}</span>\n"
        f"<span style='color:black;font-weight:bold;'>编译器分支: {branch}</span>\n"
        f"<span style='color:black;font-weight:bold;'>CommitID: {commit_id}</span>\n"
        f"---------------------------------------------------\n"
        f"<span style='color:red;font-weight:bold;'>Fail list:\n"
        f"{error_info}</span>\n"
        f"---------------------------------------------------\n"
    )
    return msg


def main():
    """ """
    if len(sys.argv) < 5:
        sys.exit(f"Usage: {sys.argv[0]} <log filepath> <recv_uid> <auth>")
        return
    log_fpath = sys.argv[1]
    receiver_uid = sys.argv[2]
    auth = sys.argv[3]
    branch = sys.argv[4]
    commit_id = sys.argv[5]
    msg = get_msg(log_fpath, branch, commit_id)
    send_msg(msg, receiver_uid, auth)


if __name__ == "__main__":
    main()
