FROM ubuntu:22.04

ARG ARCH
RUN case "${ARCH}" in \
      x86_64) echo "[INFO] Building docker image on ${ARCH}" ;; \
      aarch64) echo "[INFO] Building docker image on ${ARCH}" ;; \
      *) echo "Unsupported ARCH: ${ARCH}"; exit 1 ;; \
    esac
ENV CMAKE_VER=3.30.6
ENV TZ="Asia/shanghai"

# Install necessary tools and libraries
RUN \
  sed -i "s/ports.ubuntu.com\/ubuntu-ports/mirrors.aliyun.com\/ubuntu-ports/g" /etc/apt/sources.list; \
  apt-get update; \
  apt-get install -y wget build-essential gawk bison tmux vim git unzip zlib1g-dev python3-pip ninja-build; \
  apt-get install -y clang-15 lld-15 clang-format-15 ccache; \
  update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20; \
  update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20; \
  update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-15 20; \
  update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 20

WORKDIR /home/docker_triton-ascend_dev
COPY packages/ .
COPY setup_triton-ascend_dev.sh .
# ########## Tools
# cmake
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

# Install CANN
  chmod +x Ascend-cann-toolkit_*.run; \
  chmod +x Ascend-cann-kernels-910b_*.run; \
  ./Ascend-cann-toolkit_*.run --install --install-path=/usr/local/Ascend -q; \
  ./Ascend-cann-kernels-910b_*.run --install --install-path=/usr/local/Ascend -q;

# git
RUN \
  echo "alias ll='ls -lh'" >> $HOME/.bashrc; \
  echo "export PATH=/opt/cmake/bin:$PATH" >> $HOME/.bashrc; \
  echo "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRART_PATH" >> $HOME/.bashrc; \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
