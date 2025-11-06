#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Usage: $0 <log_filename> <op_name> <debug>"
  exit 1
fi

SCRIPT_DIR=$(readlink -f $(dirname $0))
LOG_FNAME=$1
LOG_FILE_DIR=$(readlink -f $(dirname $LOG_FNAME))
LOG_FNAME_BASE=$(basename ${LOG_FNAME%.*})
OP_NAME=$2
DEBUG=$3

# remove warnings similar to
#   [W721 02:46:23.056987540 compiler_depend.ts:159] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
sed -i -E 's/\[W721[^]]*\][^()]*\(function operator\(\)\)//g' ${LOG_FNAME}
# remove empty lines before the line containing FAILURES
FAILURES_LINE_NO=$(grep -n "FAILURES" ${LOG_FNAME} | awk -F":" '{print $1}')
if [ -n "${FAILURES_LINE_NO}" ]; then
  sed -i "1,${FAILURES_LINE_NO} { /^$/d }" ${LOG_FNAME}
  FAILURES_LINE_NO=$(grep -n "FAILURES" ${LOG_FNAME} | awk -F":" '{print $1}')
  FAILURES_LINE_NO_M1=$((FAILURES_LINE_NO - 1))
else
  WARNINGS_LINE_NO=$(grep -n "= warnings summary =" ${LOG_FNAME} | awk -F":" '{print $1}')
  if [ -n "${WARNINGS_LINE_NO}" ]; then
    sed -i "1,${WARNINGS_LINE_NO} { /^$/d }" ${LOG_FNAME}
    WARNINGS_LINE_NO=$(grep -n "= warnings summary =" ${LOG_FNAME} | awk -F":" '{print $1}')
    WARNINGS_LINE_NO_M1=$((WARNINGS_LINE_NO - 1))
  fi
fi
SCHEDULING_LINE_NO=$(grep -n "scheduling tests via LoadScheduling" ${LOG_FNAME} | awk -F":" '{print $1}')
if [ "${SCHEDULING_LINE_NO}" != "" ]; then
  # SCHEDULING_LINE_NO=$(grep -n "collecting ... collected" ${LOG_FNAME} | awk -F":" '{print $1}')
  SCHEDULING_LINE_NO_P1=$((SCHEDULING_LINE_NO + 1))
  # remove lines without [gw[0-9]]
  if [ -n "${FAILURES_LINE_NO_M1}" ]; then
    sed -i "${SCHEDULING_LINE_NO_P1},${FAILURES_LINE_NO_M1} { /\[gw[0-9]\]/!d }" ${LOG_FNAME}
  else
    if [ -n "${WARNINGS_LINE_NO_M1}" ]; then
      sed -i "${SCHEDULING_LINE_NO_P1},${WARNINGS_LINE_NO_M1} { /\[gw[0-9]\]/!d }" ${LOG_FNAME}
    # else
      # echo "[ERROR]: both 'FAILURES' and '= warnings summary =' are not found in ${LOG_FNAME}"
    fi
  fi
fi

LOG_FILE_SUMMARY_DIR=${LOG_FILE_DIR}/summary
mkdir -p ${LOG_FILE_SUMMARY_DIR}
TMP_LOG_DIR=${LOG_FILE_DIR}/tmp_log
mkdir -p ${TMP_LOG_DIR}

awk '
BEGIN { RS = "__________* test_'${OP_NAME}'"; FS = "\n" }
{
    if (NR > 1) {
        # 提取标题行（第一部分）
        title = "__________ test_'${OP_NAME}'" $1;
        print title;

        # 查找最后一个 ">" 开头的内容
        last_gt = "";
        for (i = NF; i >= 1; i--) {
            if ($i ~ /^>/) {
                last_gt = $i;
                break;
            }
        }

        # 打印最后一个 ">" 行及其后续内容（直到下一个标题或文件结束）
        if (last_gt != "") {
            print last_gt;
            for (j = i + 1; j <= NF; j++) {
                print $j;
            }
        }
        print "";  # 添加空行分隔
    }
}
' ${LOG_FNAME} | sed '/^$/N;/^\n$/D' > ${TMP_LOG_DIR}/${LOG_FNAME_BASE}_summary_tmp0.log

if [ -s "${TMP_LOG_DIR}/${LOG_FNAME_BASE}_summary_tmp0.log" ]; then
  # log file is non-empty
  ${SCRIPT_DIR}/organizeCasesByError.sh ${TMP_LOG_DIR}/${LOG_FNAME_BASE}_summary_tmp0.log > ${LOG_FILE_SUMMARY_DIR}/${LOG_FNAME_BASE}_summary.log

  ${SCRIPT_DIR}/validateNumTestCases.sh ${LOG_FNAME} ${OP_NAME} ${DEBUG}

  if [ "$DEBUG" != "1" ]; then
    rm -rf ${TMP_LOG_DIR}
  fi
fi