#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Usage: $0 <log_filename> <op_name> <debug>"
  exit 1
fi

LOG_FNAME=$1
LOG_FILE_DIR=$(readlink -f $(dirname $LOG_FNAME))
LOG_FNAME_BASE=$(basename ${LOG_FNAME%.*})
OP_NAME=$2
DEBUG=$3

SUMMARY_LINE_NO=$(grep -n "short test summary info" ${LOG_FNAME} |awk -F':' '{print $1}')
SUMMARY_LINE_NO_P1=$((SUMMARY_LINE_NO + 1))

LOG_FILE_SUMMARY_DIR=${LOG_FILE_DIR}/summary
mkdir -p ${LOG_FILE_SUMMARY_DIR}
TMP_LOG_DIR=${LOG_FILE_DIR}/tmp_log
mkdir -p ${TMP_LOG_DIR}

sed -n "${SUMMARY_LINE_NO_P1},\$p" ${LOG_FNAME} | awk '{print $2}' | awk -F'::' '{print $2}' | sed '/^$/d' | sort > ${TMP_LOG_DIR}/tmp_${LOG_FNAME_BASE}_testcases_original.txt

grep -rwni "test_${OP_NAME}" ${LOG_FILE_SUMMARY_DIR}/${LOG_FNAME_BASE}_summary.log  | awk '{print $2}' | sort > ${TMP_LOG_DIR}/tmp_${LOG_FNAME_BASE}_testcases_processed.txt

# echo "====================[INFO] test_${OP_NAME}: Compared against the original log, processing failed cases gives the following diff. None means processing works."
diff ${TMP_LOG_DIR}/tmp_${LOG_FNAME_BASE}_testcases_original.txt ${TMP_LOG_DIR}/tmp_${LOG_FNAME_BASE}_testcases_processed.txt

if [ "$DEBUG" != "1" ]; then
  rm -rf ${TMP_LOG_DIR}
fi
