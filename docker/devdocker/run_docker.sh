#!/bin/bash

ARCH=$(uname -m)

docker run \
  -v /data/disk:/home \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -e ASCEND_RUNTIME_OPTIONS=NODRV --privileged=true \
  -name triton_dev \
  -it triton-ascend_build_${ARCH}:1.0 bash