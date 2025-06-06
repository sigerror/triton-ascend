#!/bin/bash

set -ex

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

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LLVM_BUILD_DIR=/opt/llvm-b5cc222

# FIXME: 20250508 the bishengir-compile in the CANN 8.0.T115 fails lots of cases
#        So we need to use another version of compiler.
COMPILER_ROOT=/home/shared/bisheng_toolkit_20250610
export PATH=${COMPILER_ROOT}:${COMPILER_ROOT}/ccec_compiler/bin:$PATH

# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_and_test
