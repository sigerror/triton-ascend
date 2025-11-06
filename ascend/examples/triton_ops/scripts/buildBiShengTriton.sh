#!/bin/bash
# setup the env
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRART_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
. /opt/miniconda3/etc/profile.d/conda.sh
conda activate triton
# Update BiSheng-Triton
cd /root/BiSheng-Triton
git pull origin regbase
rm -rf /root/BiSheng-Triton/dist
bash scripts/build.sh $(pwd)/ascend /opt/llvm-b5cc222d7429fe6f18c787f633d5262fac2e676f/ 3.2.0 bdist_wheel false
# Copy to spcific location
mkdir -p /home/shared/Packages/BiSheng-Triton/$(date +%Y%m%d)
cp -f /root/BiSheng-Triton/dist/*.whl /home/shared/Packages/BiSheng-Triton/$(date +%Y%m%d)/
ls -alh /home/shared/Packages/BiSheng-Triton/$(date +%Y%m%d)/