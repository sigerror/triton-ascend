#!/bin/bash

docker save -o triton-dev-outofbox_latest.tar triton-dev-outofbox:latest
gzip triton-dev-outofbox_latest.tar