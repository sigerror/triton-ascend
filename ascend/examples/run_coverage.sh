#!/bin/bash
set -eo pipefail

TEST_DIR="${WORKSPACE}/triton-ascend/ascend/examples/generalization_cases"
#TEST_DIR="${WORKSPACE}/triton-ascend/ascend/examples/pytest_ut"
SRC_DIR="$(pip show triton-ascend 2>/dev/null | grep -E '^Location:' | awk -F': ' '{print $2}' | xargs)/triton"
COV_REPORT_DIR="${WORKSPACE:-./}/test_coverage"
COV_FAIL_UNDER=0
PARALLEL_WORKERS="auto"
# ==================================================

# 脚本路径/时间戳
script=$(readlink -f "$0")
script_dir=$(dirname "$script")
timestamp=$(date +%Y%m%d_%H%M%S)
cov_report_file="${COV_REPORT_DIR}/coverage_${timestamp}.xml"
cov_html_dir="${COV_REPORT_DIR}/html_report_${timestamp}"

# 日志函数
log_info() {
    echo -e "\033[32m[INFO] $(date +'%Y-%m-%d %H:%M:%S') $1\033[0m"
}

log_error() {
    echo -e "\033[31m[ERROR] $(date +'%Y-%m-%d %H:%M:%S') $1\033[0m"
    exit 1
}

# 清理缓存
clean_cache() {
    log_info "开始清理缓存文件..."
    if compgen -G "/tmp/torchinductor_*" > /dev/null; then
        rm -rf /tmp/torchinductor_* && log_info "已清理 /tmp/torchinductor_* 缓存"
    fi
    local triton_dirs=("${HOME}/.triton/dump" "${HOME}/.triton/cache")
    for dir in "${triton_dirs[@]}"; do
        [ -d "${dir}" ] && rm -rf "${dir}" && log_info "已清理 ${dir} 缓存"
    done
    log_info "缓存清理完成"
}

# 检查依赖/目录（新增源码目录检查）
check_dependencies() {
    log_info "检查必要依赖和目录..."

    # 检查pytest-cov
    if ! python -c "import pytest_cov" > /dev/null 2>&1; then
        log_info "安装pytest-cov..."
        pip install pytest-cov || log_error "pytest-cov安装失败"
    fi

    # 检查测试用例目录
    [ ! -d "${TEST_DIR}" ] && log_error "测试用例目录不存在: ${TEST_DIR}"

    # 检查业务源码目录（关键）
    [ ! -d "${SRC_DIR}" ] && log_error "业务源码目录不存在: ${SRC_DIR}"

    # 创建报告目录
    mkdir -p "${COV_REPORT_DIR}" || log_error "无法创建覆盖率报告目录: ${COV_REPORT_DIR}"
}

# 执行测试+覆盖率统计
run_test_with_coverage() {
    set -x
    log_info "===== 测试配置 ====="
    log_info "测试用例目录: ${TEST_DIR}"
    log_info "待统计覆盖率的源码目录: ${SRC_DIR}"
    log_info "覆盖率报告目录: ${COV_REPORT_DIR}"
    log_info "===================="

    pytest_args=(
        "${TEST_DIR}"
        -n "${PARALLEL_WORKERS}"
        --dist=loadfile
        --cov=triton
        --cov-report="xml:${cov_report_file}"
        --cov-report="html:${cov_html_dir}"
        --cov-fail-under="${COV_FAIL_UNDER}"
        --cov-branch
        --cov-context=test
        -v
    )
    set +e
    python3 -m pytest "${pytest_args[@]}"
    set -e

    # 验证报告生成
    [ -f "${cov_report_file}" ] || log_error "XML覆盖率报告生成失败"
    [ -d "${cov_html_dir}" ] || log_error "HTML覆盖率报告生成失败"

    log_info "XML报告路径: ${cov_report_file}"
    log_info "HTML报告路径: ${cov_html_dir}/index.html"

    # 输出覆盖率汇总
    log_info "===== 覆盖率汇总 ====="
    python -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('${cov_report_file}')
    root = tree.getroot()
    # 提取核心指标
    line_cov = float(root.attrib.get('line-rate', 0)) * 100
    branch_cov = float(root.attrib.get('branch-rate', 0)) * 100
    total_files = len(root.findall('.//class'))
    covered_lines = int(root.attrib.get('lines-covered', 0))
    total_lines = int(root.attrib.get('lines-valid', 0))

    print(f'行覆盖率: {line_cov:.2f}% ({covered_lines}/{total_lines} 行)')
    print(f'分支覆盖率: {branch_cov:.2f}%')
    print(f'统计的源码文件数: {total_files}')
except Exception as e:
    print(f'解析覆盖率报告失败: {e}')
    exit(1)
"
    log_info "======================"
}

# 主流程
main() {
    trap 'log_error "脚本执行异常中断"' ERR
    check_dependencies
    clean_cache
    run_test_with_coverage
    log_info "测试及覆盖率统计完成！"
}

# 启动
main "$@"