# docker build --build-arg ARCH=aarch64 -t triton-dev-cann:latest -f triton-dev-cann.dockerfile .
FROM triton-dev-llvm:latest

ARG ARCH
ARG CANN_VER=8.2.RC1.alpha002

COPY Ascend-cann-toolkit_${CANN_VER}_linux-${ARCH}.run ./
COPY Ascend-cann-kernels-910b_${CANN_VER}_linux-${ARCH}.run ./

RUN \
  chmod +x Ascend-cann-toolkit_${CANN_VER}_linux-${ARCH}.run; \
  chmod +x Ascend-cann-kernels-910b_${CANN_VER}_linux-${ARCH}.run; \
  ./Ascend-cann-toolkit_${CANN_VER}_linux-${ARCH}.run --install --install-path=/usr/local/Ascend -q; \
  ./Ascend-cann-kernels-910b_${CANN_VER}_linux-${ARCH}.run --install --install-path=/usr/local/Ascend -q;

RUN \
  echo "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRART_PATH" >> $HOME/.bashrc; \
  echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> $HOME/.bashrc

# clean
RUN \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* Ascend-cann-toolkit_${CANN_VER}_linux-${ARCH}.run Ascend-cann-kernels-910b_${CANN_VER}_linux-${ARCH}.run
