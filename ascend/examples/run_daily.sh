#!/bin/bash

set -ex

script=$(readlink -f "$0")
script_dir=$(dirname "$script")

export LLVM_BUILD_DIR=/triton_depends/llvm-20-install

source /usr/local/CANN_8.2.RC1.alpha002/ascend-toolkit/set_env.sh
export LLVM_BUILD_DIR=/opt/llvm-b5cc222

COMPILER_ROOT=/home/shared/bisheng_toolkit_20250628
export PATH=${COMPILER_ROOT}:${COMPILER_ROOT}/8.2.RC1.alpha002/compiler/bishengir/bin:$PATH


function build_triton() {

  cd ${WORKSPACE}
  # Run uninstall once because the while-loop does not stop. No idea why.
  # uninstall_triton_ascend
  pip3 uninstall triton_ascend -y

  git submodule set-url third_party/triton https://gitee.com/shijingchang/triton.git
  git submodule sync && git submodule update --init --recursive

  bash scripts/build.sh ${WORKSPACE}/ascend ${LLVM_BUILD_DIR} 3.2.0 install 0
}


# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_triton

if [ -d ${WORKSPACE}triton ];then
  rm -rf ${WORKSPACE}triton
fi

if [ -d ~/.triton/dump ];then
  rm -rf ~/.triton/dump
fi

if [ -d ~/.triton/cache ];then
  rm -rf ~/.triton/cache
fi

cd ${WORKSPACE}

# 定义日志文件名（格式示例：test_results_20231015_153045.log）
GENE_CASE_LOG_FILE="${WORKSPACE}/test_generalizetion_case_$(date +%Y%m%d).log"

# 定义要运行的测试目录
TEST_generalization="${WORKSPACE}/ascend/examples/generalization_cases"

# 记录测试开始时间
echo "===== 测试开始时间: $(date +"%Y-%m-%d %H:%M:%S") =====" > "$GENE_CASE_LOG_FILE"

#函数:运行测试并返回状态码
function run_tests() {
  local dir="$1"
  echo "Running tests in dir: $dir" | tee -a "$GENE_CASE_LOG_FILE"
  cd ${dir} || {
    echo "failed to cd to : ${dir}" | tee -a "$GENE_CASE_LOG_FILE"
    return 1
    }

  #运行测试
  pytest -sv -n 16 . 2>&1 | tee -a "$GENE_CASE_LOG_FILE"

  local pytest_exit=$?
  return $pytest_exit
}
cd ${WORKSPACE}
run_tests "${TEST_generalization}"

echo -e "\n===== 测试结束时间: $(date +"%Y-%m-%d %H:%M:%S") =====" >> "$GENE_CASE_LOG_FILE"
cp "$GENE_CASE_LOG_FILE" "/home/daily_log"

# run inductor cases
TEST_inductor_cases="${WORKSPACE}/ascend/examples/inductor_cases"
cd ${TEST_inductor_cases}
bash run_inductor_test.sh

#run flaggems cases
TEST_flaggems_cases="${WORKSPACE}/ascend/examples/flaggems_cases"
cd ${TEST_flaggems_cases}
bash run_flaggems_test.sh

