#!/bin/bash

set -ex

inductor_skip_list=(
  "test_check_accuracy.py"
  "test_debug_msg.py"
  "test_embedding.py"
  "test_force_fallback.py"
  "test_foreach_add.py"
  "test_geometric.py"
  "test_lazy_register.py"
  )

function uninstall_triton_ascend() {
  set +e
  while true; do
    pip3 uninstall triton_ascend -y | grep "Found existing installation"
    if [ $? -eq 1 ]; then
        echo "All triton_ascend versions are uninstalled"
        break
    fi
  done
  set -e
}

function build_and_test() {
  if [ -d ${HOME}/.triton/dump ];then
    rm -rf ${HOME}/.triton/dump
  fi
  if [ -d ${HOME}/.triton/cache ];then
    rm -rf ${HOME}/.triton/cache
  fi

  cd ${WORKSPACE}
  # Run uninstall once because the while-loop does not stop. No idea why.
  # uninstall_triton_ascend
  pip3 uninstall triton_ascend -y

  git submodule set-url third_party/triton https://gitee.com/shijingchang/triton.git
  git submodule sync && git submodule update --init --recursive

  bash scripts/build.sh ${WORKSPACE}/ascend ${LLVM_BUILD_DIR} 3.2.0 install 0

  TEST_triton="${WORKSPACE}/ascend/examples/pytest_ut"
  cd ${TEST_triton}
  pytest -n 16 --dist=load . || { exit 1 ; }

  TEST_inductor="${WORKSPACE}/ascend/examples/inductor_cases"
  cd ${TEST_inductor}
  git init
  git remote add origin http://gitee.com/ascend/pytorch.git
  git config core.sparsecheckout true
  echo "test/_inductor" >> .git/info/sparse-checkout
  git pull origin v2.6.0:master
  TEST_inductor_cases_path="${TEST_inductor}/test/_inductor"
  cd ${TEST_inductor_cases_path}
  export PYTHONPATH="${PYTHONPATH}:${TEST_inductor_cases_path}"
  for skip_case in ${inductor_skip_list[@]};
  do
    if [ -e "${TEST_inductor_cases_path}/${skip_case}" ];then
      echo "skip test case of ${skip_case}"
      mv ${skip_case} "${skip_case}_skip"
    fi
  done
  pytest -n 16 --dist=load . || { exit 1 ; }
}

function validate_git_commit_title() {
  if [ $# -lt 1 ]; then
    echo "Usage: $0 <commit_title>"
    exit 1
  fi
  commit_title=$1
  if ! echo "${commit_title}" | grep -qE "^(feat|fix|docs|style|refactor|test|chore|revert)(\(.*\))?: .+"; then
    echo "❌ The git commit title does not comply with the specifications!"
    echo "Format Requirements: <type>(<scope>): <subject>"
    echo "e.g.: feat(user): The login function is added."
    echo "Allowed Types: feat | fix | docs | style | refactor | test | chore | revert"
    exit 1
  fi
  echo "✅ The submitted information complies with the specifications."
}

function validate_pr_all_commits_title() {
  commit_titles=$(git log master..HEAD --oneline | sed 's/^[^ ]* //')
  if [ -z "$commit_titles" ]; then
    echo "No commits found between HEAD and master."
    exit 1
  fi
  echo "Validating commit titles..."
  echo "----------------------------"
  while IFS= read -r title; do
    echo "Checking: $title"
    if ! validate_git_commit_title "$title" 2>/dev/null; then
      echo "Error in commit: $title" >&2
      HAS_ERROR=true
    fi
  done <<< "$commit_titles"
  if [ "$HAS_ERROR" = true ]; then
    echo "----------------------------"
    echo "❌ Some commit titles do not meet the specifications." >&2
    exit 1
  else
    echo "----------------------------"
    echo "✅ All commit titles meet the specifications."
  fi
}

# if ! validate_pr_all_commits_title 2>/dev/null; then
#   exit 1
# fi

source /usr/local/CANN_8.2.RC1.alpha002/ascend-toolkit/set_env.sh
export LLVM_BUILD_DIR=/opt/llvm-b5cc222

# FIXME: 20250508 the bishengir-compile in the CANN 8.0.T115 fails lots of cases
#        So we need to use another version of compiler.
COMPILER_ROOT=/home/shared/bisheng_toolkit_20250628
export PATH=${COMPILER_ROOT}:${COMPILER_ROOT}/8.2.RC1.alpha002/compiler/bishengir/bin:$PATH

# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_and_test
