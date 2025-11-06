# docker build --build-arg PROXY_URL=http://工号:转义后的密码@proxyhk.huawei.com:8080 -t triton-dev-triton:latest -f triton-dev-triton.dockerfile .

FROM triton-dev-cann:latest

ARG PROXY_URL

ENV http_proxy=$PROXY_URL
ENV https_proxy=$PROXY_URL

# conda
RUN \
  . /opt/miniconda3/etc/profile.d/conda.sh && \
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
  conda create -n triton python=3.11 -y && \
  /opt/miniconda3/bin/conda clean -afy
ENV http_proxy= https_proxy=
# torchnpu
RUN \
  . /opt/miniconda3/etc/profile.d/conda.sh && \
  conda activate triton; \
  pip3 install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11 torch_npu==2.6.0rc1
# env
RUN \
  git config --global http.sslVerify false; \
  ln -s /home/shared/configs/tmux.conf $HOME/.tmux.conf; \
  ln -s /home/shared/configs/tmux $HOME/.tmux; \
  mv /opt/miniconda3/envs/triton/lib/libstdc++.so.6 /opt/miniconda3/envs/triton/lib/libstdc++.so.6.backup

# clean
RUN \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
