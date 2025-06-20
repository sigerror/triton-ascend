FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"

RUN apt-get update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    bash \
    ca-certificates \
    curl \
    git \
    gnupg \
    sudo \
    unzip \
    vim \
    wget && \
    update-ca-certificates

RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 && \
    apt-get update

RUN apt install --yes --no-install-recommends --no-install-suggests \
    clang \
    clang-format \
    cmake \
    lld \
    ninja-build \
    python3.10 \
    python3.10-distutils \
    python3.11 \
    python3.11-distutils \
    python3.9 \
    python3.9-distutils \
    zlib1g-dev
	
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 --slave /usr/bin/pip pip /usr/local/bin/pip3.9 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2 --slave /usr/bin/pip pip /usr/local/bin/pip3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 3 --slave /usr/bin/pip pip /usr/local/bin/pip3.11 && \
    rm -f /usr/local/bin/pip /usr/local/bin/pip3 && \
    update-alternatives --set python /usr/bin/python3.11 && \
    ln -s /usr/bin/pip /usr/bin/pip3

RUN curl -sS https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C22B800TP026/Ascend-cann-toolkit_8.2.RC1.alpha002_linux-$(uname -i).run -o Ascend-cann-toolkit.run && \
    curl -sS https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C22B800TP026/Ascend-cann-kernels-910b_8.2.RC1.alpha002_linux-$(uname -i).run -o Ascend-cann-kernels.run && \
    chmod +x Ascend-cann-toolkit.run && \
    chmod +x Ascend-cann-kernels.run && \
    ./Ascend-cann-toolkit.run --full --quiet && \
    ./Ascend-cann-kernels.run --install --quiet && \
    rm -f Ascend-cann-toolkit.run Ascend-cann-kernels.run

RUN useradd triton \
    --create-home \
    --shell=/bin/bash \
    --uid=1000 \
    --user-group && \
    echo "triton ALL=(ALL) NOPASSWD:ALL" >>/etc/sudoers.d/nopasswd && \
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /home/triton/.bashrc
	
USER triton

WORKDIR /home/triton

