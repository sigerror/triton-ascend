# First prepare files: id_rsa, bisheng, ld.lld, bishengir-compile in this directory
# docker build --build-arg PROXY_URL=http://工号:转义后的密码@proxyhk.huawei.com:8080 -t triton-dev-outofbox:latest -f triton-dev-outofbox.dockerfile .

FROM triton-dev-triton:latest

ARG PROXY_URL
ARG TARGET_COMMIT=b5cc222d7429fe6f18c787f633d5262fac2e676f

COPY . .

# env
RUN \
  mkdir $HOME/.ssh && \
  chmod 700 $HOME/.ssh && \
  mv id_rsa $HOME/.ssh/ && \
  chmod 600 $HOME/.ssh/id_rsa && \
  ssh-keyscan -p 2222 codehub-dg-y.huawei.com >> $HOME/.ssh/known_hosts && \
  git config --global user.email "shijingchang@huawei.com" && \
  git config --global user.name "s00653124" && \
  BISHENGIR_TOOLKIT_DIR=/opt/bishengir_toolkit_regbase && \
  mkdir -p ${BISHENGIR_TOOLKIT_DIR}/ccec_compiler/bin && \
  mv bisheng ${BISHENGIR_TOOLKIT_DIR}/ccec_compiler/bin/ && \
  mv ld.lld ${BISHENGIR_TOOLKIT_DIR}/ccec_compiler/bin/ && \
  mv bishengir-compile ${BISHENGIR_TOOLKIT_DIR}/ && \
  echo "export PATH=${BISHENGIR_TOOLKIT_DIR}:${BISHENGIR_TOOLKIT_DIR}/ccec_compiler/bin:$PATH" >> $HOME/.bashrc

# BiSheng-Triton
RUN \
  cd $HOME && \
  git clone ssh://git@codehub-dg-y.huawei.com:2222/CompilerKernel/MatrixCompiler/BiSheng-Triton.git -b regbase
ENV http_proxy=$PROXY_URL
ENV https_proxy=$PROXY_URL
RUN \
  cd $HOME && \
  cd BiSheng-Triton && \
  git submodule update --init && \
  . /opt/miniconda3/etc/profile.d/conda.sh && \
  conda activate triton && \
  export PATH=/opt/cmake/bin:$PATH && \
  bash scripts/build.sh $(pwd)/ascend /opt/llvm-${TARGET_COMMIT} 3.2.0 develop true
ENV http_proxy= https_proxy=

# pip
RUN \
  . /opt/miniconda3/etc/profile.d/conda.sh && \
  conda activate triton && \
  pip3 install paramiko && \
  conda config --set auto_activate_base false && \
  echo "conda activate triton" >> $HOME/.bashrc

# clean
RUN \
  git config --global --unset user.email && \
  git config --global --unset user.name && \
  rm $HOME/.ssh/id_rsa && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /root/BiSheng-Triton
