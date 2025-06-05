#!/bin/bash
set -e

WORK_ROOT=$(pwd)
SHARED_DIR=/home/shared
mkdir -P ${SHARED_DIR}

# TODO: check this file exists
source /opt/miniconda3/etc/profile.d/conda.sh

declare -A py_map=(["py39"]="3.9" ["py310"]="3.10" ["py311"]="3.11")

for env_name in "${!py_map[@]}"; do
    if ! conda env list | grep -wq "$env_name"; then
      conda create -y -n "$env_name" python="${py_map[$env_name]}"
      mv /opt/miniconda3/envs/"$env_name"/lib/libstdc++.so.6 /opt/miniconda3/envs/"$env_name"/lib/libstdc++.so.6.bak
    fi
    conda activate $env_name
    pip3 install pybind11 torch_npu==2.6.0rc1 numpy==1.26.4 scipy attrs decorator psutil pytest pytest-xdist pyyaml
done

# LLVM
cd $SHARED_DIR
if [ ! -d llvm-project ]; then
  git clone --depth 1 https://github.com/llvm/llvm-project.git
fi
cd llvm-project
TARGET_COMMIT="b5cc222d7429fe6f18c787f633d5262fac2e676f"
if ! git rev-parse --quiet --verify "$TARGET_COMMIT" &>/dev/null; then
  echo "Error: Commit $TARGET_COMMIT does not exist!" >&2
  exit 1
fi
if [ "$(git rev-parse HEAD)" != "$TARGET_COMMIT" ]; then
  echo "LLVM commit is NOT $TARGET_COMMIT"
  git fetch --depth 1 origin b5cc222d7429fe6f18c787f633d5262fac2e676f
  git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
else
  echo "LLVM commit is $TARGET_COMMIT"
fi
if [ ! -d /opt/llvm-b5cc222/lib ]; then
  mkdir -p build
  cd build
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm-b5cc222 \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON
  ninja install
fi

cd $WORK_ROOT
if [ ! -d "triton-ascend" ]; then
  git clone --recurse-submodules https://gitee.com/ascend/triton-ascend.git
fi

cd triton-ascend

environments=("py39" "py310" "py311")

for env in "${environments[@]}"; do
    (
        conda activate "$env" && \
        echo "在 $env 中执行命令" && \
        bash scripts/build.sh $(pwd)/ascend /opt/llvm-b5cc222 ${TRITON_VERSION} install 0
    ) || echo "$env 执行失败"
done
