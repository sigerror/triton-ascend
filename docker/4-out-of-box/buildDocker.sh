#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <proxy_url>"
  echo "proxy_url should be like http://工号:转义后的密码@proxyhk.huawei.com:8080"
  exit 1
fi

proxy_url=$1

REQUIRED_FILE=bisheng
if [ ! -e ${REQUIRED_FILE} ]; then
  echo "[ERROR]: ${REQUIRED_FILE} does not exist in the current directory!"
  exit 1
fi

REQUIRED_FILE=bishengir-compile
if [ ! -e ${REQUIRED_FILE} ]; then
  echo "[ERROR]: ${REQUIRED_FILE} does not exist in the current directory!"
  exit 1
fi

REQUIRED_FILE=ld.lld
if [ ! -e ${REQUIRED_FILE} ]; then
  echo "[ERROR]: ${REQUIRED_FILE} does not exist in the current directory!"
  exit 1
fi

REQUIRED_FILE=id_rsa
if [ ! -e ${REQUIRED_FILE} ]; then
  echo "[ERROR]: ${REQUIRED_FILE} does not exist in the current directory!"
  echo "         id_rsa must be provided to have permission to pull from codehub!"
  exit 1
fi

docker build --build-arg PROXY_URL=${proxy_url} -t triton-dev-outofbox:latest -f triton-dev-outofbox.dockerfile .
