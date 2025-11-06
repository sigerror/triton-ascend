# docker build --build-arg ARCH=aarch64 --build-arg PROXY_URL=http://工号:转义后的密码@proxyhk.huawei.com:8080 -t triton-dev-base:latest -f triton-dev-base.dockerfile .

FROM ubuntu:22.04

ARG ARCH
ARG PROXY_URL

WORKDIR /home

RUN case "${ARCH}" in \
      x86_64) echo "[INFO] Building docker image on ${ARCH}" ;; \
      aarch64) echo "[INFO] Building docker image on ${ARCH}" ;; \
      *) echo "Unsupported ARCH: ${ARCH}"; exit 1 ;; \
    esac
ENV CMAKE_VER=3.30.6
ENV TZ="Asia/shanghai"

# Install necessary tools and libraries
RUN \
  sed -i "s/ports.ubuntu.com\/ubuntu-ports/mirrors.tools.huawei.com\/ubuntu-ports/g" /etc/apt/sources.list; \
  apt-get update; \
  apt-get install -y wget curl; \
  apt-get install -y build-essential gawk bison tmux vim git unzip zlib1g-dev python3-pip ninja-build; \
  apt-get install -y clang-15 lld-15 clang-format-15 ccache; \
  update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20; \
  update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20; \
  update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-15 20; \
  update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 20

# ########## Tools
# cmake
ENV http_proxy=$PROXY_URL
ENV https_proxy=$PROXY_URL
RUN \
  if [ ! -f "cmake-${CMAKE_VER}-linux-${ARCH}.sh" ]; then \
    wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-${ARCH}.sh; \
  fi; \
  chmod +x cmake-${CMAKE_VER}-linux-${ARCH}.sh; \
  mkdir -p /opt/cmake; \
  ./cmake-${CMAKE_VER}-linux-${ARCH}.sh --skip-license --prefix=/opt/cmake
# conda
RUN \
  if [ ! -f "Miniconda3-latest-Linux-${ARCH}.sh" ]; then \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh; \
  fi; \
  chmod +x Miniconda3-latest-Linux-${ARCH}.sh; \
  ./Miniconda3-latest-Linux-${ARCH}.sh -bfs -p /opt/miniconda3; \
  echo ". /opt/miniconda3/etc/profile.d/conda.sh" >> $HOME/.bashrc; \
  /opt/miniconda3/bin/conda clean -afy
ENV http_proxy= https_proxy=
# env
RUN \
  echo "alias ll='ls -lh'" >> $HOME/.bashrc; \
  echo "export PATH=/opt/cmake/bin:$PATH" >> $HOME/.bashrc; \
  pip3 config set global.index-url http://mirrors.tools.huawei.com/pypi/simple; \
  pip3 config set global.trusted-host mirrors.tools.huawei.com

# clean
RUN \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* cmake-${CMAKE_VER}-linux-${ARCH}.sh Miniconda3-latest-Linux-${ARCH}.sh
