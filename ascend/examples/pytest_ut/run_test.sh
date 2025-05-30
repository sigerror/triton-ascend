#!/bin/bash

set -ex

function build_and_test() {
  if [ -d ${HOME}/.triton/dump ];then
    rm -rf ${HOME}/.triton/dump
  fi
  if [ -d ${HOME}/.triton/cache ];then
    rm -rf ${HOME}/.triton/cache
  fi

  cd ${WORKSPACE}
  git submodule set-url third_party/triton https://gitee.com/shijingchang/triton.git
  git submodule sync && git submodule update --init --recursive

  LLVM_SYSPATH=${LLVM_BUILD_DIR} \
  TRITON_PLUGIN_DIRS=${WORKSPACE}/ascend \
  TRITON_WHEEL_NAME="triton_ascend" \
  TRITON_VERSION=3.2.0 \
  TRITON_BUILD_WITH_CCACHE=true \
  TRITON_BUILD_WITH_CLANG_LLD=true \
  TRITON_BUILD_PROTON=OFF \
  TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
  python3 setup.py install

  cd ${WORKSPACE}/ascend/examples/pytest_ut
  pytest -n 16 --dist=load . || { exit 1 ; }
}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LLVM_BUILD_DIR=/opt/llvm-b5cc222

# FIXME: 20250508 the bishengir-compile in the CANN 8.0.T115 fails lots of cases
#        So we need to use another version of compiler.
COMPILER_ROOT=/home/shared/bisheng_toolkit_20250519
export PATH=${COMPILER_ROOT}:${COMPILER_ROOT}/ccec_compiler/bin:$PATH

# build in torch 2.3.1
source /opt/miniconda3/bin/activate torch_231
build_and_test

# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_and_test
