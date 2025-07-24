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

function run_case_by_multi_card() {

    EACH_DEVICE_THREAD=64
    # 获取可用昇腾卡数量
    NPU_DEVICES=$(ls /dev/davinci? 2>/dev/null | wc -l)
    [ $NPU_DEVICES -eq 0 ] && {
        echo "No Ascend devices found!"
        exit 1
    }

    echo "Detected $NPU_DEVICES Ascend devices"

    if [ -d ${WORKSPACE}triton ];then
      rm -rf ${WORKSPACE}triton
    fi

    if [ -d ~/.triton/dump ];then
      rm -rf ~/.triton/dump
    fi

    if [ -d ~/.triton/cache ];then
      rm -rf ~/.triton/cache
    fi

    test_dir=$1

    cd ${test_dir}

    # 清理旧日志
    rm -rf logs && mkdir logs

    # 记录测试开始时间
    echo "===== 测试开始时间: $(date +"%Y-%m-%d %H:%M:%S") ====="
    n_value=$(echo "scale=0; $NPU_DEVICES * $EACH_DEVICE_THREAD * 0.75 / 1" | bc)

    pytest ${test_dir} -n ${n_value} --dist=loadscope -v --junitxml=logs/results.xml | tee logs/raw_output.log

    # 处理日志（添加设备标签）
    awk '
      />> Worker gw[0-9]+ using NPU device/ {
        split($0, parts, / /)
        dev_id = parts[NF]
        worker = parts[3]
        print "[" strftime("%Y-%m-%d %H:%M:%S") "| DEV-" dev_id "] " $0
        next
      }
      { print "[" strftime("%Y-%m-%d %H:%M:%S") "| DEV-" dev_id "] " $0 }
    ' logs/raw_output.log > logs/combined.log

    echo "========================================"
    echo "All tests completed!"
    echo "JUnit Report: logs/results.xml"
    echo "Combined Log: logs/combined.log"
    echo "========================================"

    echo -e "\n===== 测试结束时间: $(date +"%Y-%m-%d %H:%M:%S") ====="

    log_file=$2
    cp ${test_dir}/logs/combined.log "/home/daily_log/${log_file}"
}

# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_triton

cd ${WORKSPACE}

# run gene case
log_file="test_generalizetion_case_$(date +%Y%m%d).log"
TEST_generalization="${WORKSPACE}/ascend/examples/generalization_cases"
run_case_by_multi_card ${TEST_generalization} ${log_file}

# run inductor cases
TEST_inductor_cases="${WORKSPACE}/ascend/examples/inductor_cases"
cd ${TEST_inductor_cases}
bash run_inductor_test.sh

# run flaggems cases
TEST_flaggems_cases="${WORKSPACE}/ascend/examples/flaggems_cases"
cd ${TEST_flaggems_cases}
bash run_flaggems_test.sh



