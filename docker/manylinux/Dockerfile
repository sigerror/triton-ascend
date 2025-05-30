ARG ARCH="aarch64"
FROM quay.io/pypa/manylinux_2_28_${ARCH}:latest

# ARG Definition
ARG ARCH
ARG PYTHON_VERSION
ARG LLVM_PATH
ARG LLVM_COMMITID
ENV PYTHON_PATH=/opt/python/cp${PYTHON_VERSION}-cp${PYTHON_VERSION}
ENV NINJA_PATH=https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux

WORKDIR /root

# YUM
RUN yum clean all && yum makecache && yum install -y vim wget clang cmake ccache unzip lld

# Python
RUN ln -s ${PYTHON_PATH}/bin/python3 /usr/local/bin/python3 && \
    ln -s ${PYTHON_PATH}/bin/python3 /usr/local/bin/python && \
    ln -s ${PYTHON_PATH}/bin/pip3 /usr/local/bin/pip3 && \
    pip3 install ninja cmake wheel auditwheel patchelf pybind11 twine

# Ninja
RUN wget --no-check-certificate ${NINJA_PATH}$([ "${ARCH}" = "aarch64" ] && echo "-aarch64" || echo "").zip -O ninja-linux.zip && \
    unzip ninja-linux.zip && \
    mv ninja /usr/local/bin && \
    rm -f ninja-linux.zip

# Context for Speedup
COPY . /context

# LLVM Codebase
RUN if [ -n "${LLVM_PATH}" ]; then mv /context/${LLVM_PATH} llvm-project; else git clone https://github.com/llvm/llvm-project.git; fi

# LLVM Installation
RUN pushd llvm-project && \
    git checkout ${LLVM_COMMITID} && \
    git reset --hard HEAD && \
    git submodule sync && git submodule update --init --recursive && \
    git clean -dfx && git submodule foreach git clean -dfx && \
    mkdir build && cd build && \
    cmake -G Ninja ../llvm -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DCMAKE_INSTALL_PREFIX=/usr/local/llvm && \
    ninja -j 32 install && \
    popd && rm -rf llvm-project

RUN echo "export IS_MANYLINUX=True" > /root/.bashrc
