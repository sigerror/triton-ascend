#!/bin/bash
recv_gid=791007509686292884
auth=y30015927_RZp3HeFUIvb7hyGTks0SC1WMPlYfxcmX
# setup the env
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRART_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
. /opt/miniconda3/etc/profile.d/conda.sh
conda activate triton
# Choose selected branch: feature-bishengir-regbase(default) / dev-19x 
if [ $# -eq 0 ]; then
    BRANCH="feature-bishengir-regbase"
else
    BRANCH="$1"
fi
# Update BiShengIR
cd /home/shared/BiShengIR/build
git checkout -b $BRANCH 
git pull origin $BRANCH
commit_id=$(git rev-parse HEAD)
ninja install
cp -f /home/shared/BiShengIR/build/install/bin/bishengir-compile /opt/bishengir_toolkit_regbase/bishengir-compile
# Run Triton tests
COMPILER_ROOT=/opt/bishengir_toolkit_regbase
export PATH=$COMPILER_ROOT:$COMPILER_ROOT/ccec_compiler/bin:$PATH
ARCHIVE_DIR=/home/shared/triton-regbase-daily-tests
TEST_GROUP=simple
date_str=$(date +%Y%m%d)
if [ -d ${ARCHIVE_DIR}/${date_str} ]; then
  # We previously run cases at the same day. So we backup the generated one.
  mv ${ARCHIVE_DIR}/${date_str} ${ARCHIVE_DIR}/${date_str}_$(printf "%08x" $RANDOM$RANDOM$RANDOM | cut -c1-8)
fi
mkdir ${ARCHIVE_DIR}/$(date +%Y%m%d)
# Run 730 tests 
TARGET_OPS_GROUP_NAME=730
bash /root/BiSheng-Triton/ascend/examples/triton_ops/scripts/runCases.sh ${TARGET_OPS_GROUP_NAME} ${TEST_GROUP} ${ARCHIVE_DIR} 2>&1 | tee ${ARCHIVE_DIR}/$(date +%Y%m%d)/${TARGET_OPS_GROUP_NAME}.log
# Send message to Welink group: BiShengMAX Regbase
python3 /root/BiSheng-Triton/ascend/examples/triton_ops/scripts/sendMsgToWelink.py ${ARCHIVE_DIR}/$(date +%Y%m%d)/${TARGET_OPS_GROUP_NAME}.log ${recv_gid} ${auth} ${BRANCH} ${commit_id}
# Run 830 tests 
TARGET_OPS_GROUP_NAME=830
bash /root/BiSheng-Triton/ascend/examples/triton_ops/scripts/runCases.sh ${TARGET_OPS_GROUP_NAME} ${TEST_GROUP} ${ARCHIVE_DIR} 2>&1 | tee ${ARCHIVE_DIR}/$(date +%Y%m%d)/${TARGET_OPS_GROUP_NAME}.log
python3 /root/BiSheng-Triton/ascend/examples/triton_ops/scripts/sendMsgToWelink.py ${ARCHIVE_DIR}/$(date +%Y%m%d)/${TARGET_OPS_GROUP_NAME}.log ${recv_gid} ${auth} ${BRANCH} ${commit_id}
# Run 930 tests 
TARGET_OPS_GROUP_NAME=930
bash /root/BiSheng-Triton/ascend/examples/triton_ops/scripts/runCases.sh ${TARGET_OPS_GROUP_NAME} ${TEST_GROUP} ${ARCHIVE_DIR} 2>&1 | tee ${ARCHIVE_DIR}/$(date +%Y%m%d)/${TARGET_OPS_GROUP_NAME}.log
python3 /root/BiSheng-Triton/ascend/examples/triton_ops/scripts/sendMsgToWelink.py ${ARCHIVE_DIR}/$(date +%Y%m%d)/${TARGET_OPS_GROUP_NAME}.log ${recv_gid} ${auth} ${BRANCH} ${commit_id}