#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Usage: $0 [730|830] [simple|full] <directory to hold the results>"
  exit 1
fi
# Allowed values: [730, 830]
TARGET_OPS_GROUP_NAME=$1
# Allowed values: [simple, full]
TEST_GROUP=$2
# Normally we place the results in /home/shared/triton-regbase-daily-tests
ARCHIVE_DIR=$3

NUM_THREADS=7
DEBUG=1

SCRIPT_DIR=$(readlink -f $(dirname $0))
RESULT_DIR_NAME=${TARGET_OPS_GROUP_NAME}_log_cache
if ! python3 -c "import sys, json; json.load(sys.stdin)" < ${SCRIPT_DIR}/test_ops.json; then
  echo "[ERROR]: ${SCRIPT_DIR}/test_ops.json is invalid!"
fi
TARGET_OPS=$(python3 ${SCRIPT_DIR}/parseTestOps.py ${SCRIPT_DIR}/test_ops.json ${TARGET_OPS_GROUP_NAME})

OPS_ROOT_DIR=$(dirname ${SCRIPT_DIR})
RESULT_DIR=${SCRIPT_DIR}/${RESULT_DIR_NAME}
mkdir -p ${RESULT_DIR}
rm -rf ${SCRIPT_DIR}/cache ~/.triton/remote_cache/ ~/.triton/dump/

total_ops=$(echo "$TARGET_OPS" | wc -w)

current_op=0
for op_and_file in ${TARGET_OPS}; do
  op=$(echo "$op_and_file" | cut -d',' -f1)
  op_file=$(echo "$op_and_file" | cut -d',' -f2)
  ((current_op++))
  percent=$((current_op * 100 / total_ops))
  bar_length=50
  filled_length=$((percent * bar_length / 100))
  empty_length=$((bar_length - filled_length))
  filled_bar=$(printf "%${filled_length}s" | tr ' ' '#')
  empty_bar=$(printf "%${empty_length}s" | tr ' ' '-')
  echo -ne "\r[${filled_bar}${empty_bar}] ${percent}% ($current_op/$total_ops) - Currnet Op: $op\n"

  test_file_path=${OPS_ROOT_DIR}/${op_file}
  PYTHONPATH=${OPS_ROOT_DIR}:$PYTHONPATH \
  TRITON_CACHE_DIR=${SCRIPT_DIR}/cache \
  TRITON_ASCEND_ARCH=Ascend310B4 \
  TRITON_ENABLE_TASKQUEUE=0 \
  pytest -n ${NUM_THREADS} --dist=load \
  -sv ${test_file_path} 2>&1 | tee "${RESULT_DIR}/${op}.log"
  if [ "${DEBUG}" == "1" ]; then
    mv ${SCRIPT_DIR}/cache ${RESULT_DIR}/cache_${op}
  else
    rm -rf ${SCRIPT_DIR}/cache
  fi
done

echo -e "\r[##################################################] 100% ($total_ops/$total_ops) - All ops are run\r"

current_op=0
for op_and_file in ${TARGET_OPS}; do
  op=$(echo "$op_and_file" | cut -d',' -f1)
  op_file=$(echo "$op_and_file" | cut -d',' -f2)
  ((current_op++))
  percent=$((current_op * 100 / total_ops))
  bar_length=50
  filled_length=$((percent * bar_length / 100))
  empty_length=$((bar_length - filled_length))
  filled_bar=$(printf "%${filled_length}s" | tr ' ' '#')
  empty_bar=$(printf "%${empty_length}s" | tr ' ' '-')
  echo -ne "\r[${filled_bar}${empty_bar}] ${percent}% ($current_op/$total_ops) - Currnet Op: $op\n"

  output=$("${SCRIPT_DIR}/postProcessTestLog.sh" "${RESULT_DIR}/${op}.log" "${op}" "${DEBUG}" 2>&1)

  if [ -n "$output" ]; then
      echo "[ERROR]: ${op} fails with the following output" >&2
      echo "$output" >&2
  fi
done

echo -e "\r[##################################################] 100% ($total_ops/$total_ops) - All ops are post-processed\r"

python3 ${SCRIPT_DIR}/countNumPassedFailedCases.py ${RESULT_DIR}

python3 ${SCRIPT_DIR}/statErrors.py ${RESULT_DIR}/summary

date_str=$(date +%Y%m%d)
mkdir -p ${ARCHIVE_DIR}/${date_str}
mv -f ${RESULT_DIR} ${ARCHIVE_DIR}/${date_str}/
echo "${RESULT_DIR} is saved to ${ARCHIVE_DIR}/${date_str}/"
