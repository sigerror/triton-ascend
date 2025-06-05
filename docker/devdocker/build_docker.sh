#!/bin/bash

ARCH=$(uname -m)
docker build --build-arg ARCH=${ARCH} -t triton-ascend_dev_${ARCH}:1.0 -f triton-ascend_dev.dockerfile .