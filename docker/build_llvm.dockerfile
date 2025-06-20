#Prase 1, Download llvm
FROM ubuntu AS checkout_repo

RUN apt-get update && \
    apt-get install --yes --no-install-recommends --no-install-suggests \
    git \
    ca-certificates \
    && \
    update-ca-certificates

COPY llvm-hash.txt .

RUN LLVM_COMMIT=$(cat llvm-hash.txt) && \ 
    git clone --no-checkout https://github.com/llvm/llvm-project.git && \
    cd llvm-project && \
    git checkout $LLVM_COMMIT && \
    cd ..


# Phase 2.1, build with ubuntu 22.04
FROM ubuntu:22.04 AS ubuntu_builder

# Install build dependencies
RUN apt-get update && \
    apt-get install --yes --no-install-recommends --no-install-suggests \
    ccache \
    clang \
    ninja-build \
    lld \
    git \
    python3 \
    python3-dev \
    python3-pip && \
    python3 -m pip install cmake ninja

COPY --from=checkout_repo llvm-project llvm-project

# Install MLIR's Python Dependencies
RUN python3 -m pip install -r llvm-project/mlir/python/requirements.txt

# Configure, Build and Install LLVM
RUN SHORT_LLVM_COMMIT_HASH="$(git -C llvm-project rev-parse --short=8 HEAD)" && \
    ARCH=$(uname -i) && \
    case "$ARCH" in \
      x86_64) ARCH_NAME="x64" ;; \
      aarch64) ARCH_NAME="arm64" ;; \
      *) ARCH_NAME="$ARCH" ;; \
    esac && \
    INSTALL_DIR="llvm-${SHORT_LLVM_COMMIT_HASH}-ubuntu-${ARCH_NAME}" && \
    cmake -GNinja -Bllvm-project/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_ASM_COMPILER=clang \
    -DCMAKE_LINKER=lld \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_ENABLE_DIA_SDK=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_TERMINFO=OFF \
    llvm-project/llvm && \
    ninja -C llvm-project/build install && \
    tar czf ${INSTALL_DIR}.tar.gz ${INSTALL_DIR}

# Phase 2.2 build with almalinux
FROM almalinux:8 AS almalinux_builder

RUN dnf install --assumeyes llvm-toolset
RUN dnf install --assumeyes python38-pip python38-devel git

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install --upgrade cmake ninja lit

COPY --from=checkout_repo llvm-project llvm-project

# Install MLIR's Python Dependencies
RUN python3.8 -m pip install -r llvm-project/mlir/python/requirements.txt

# Configure, Build, and Install LLVM
RUN SHORT_LLVM_COMMIT_HASH="$(git -C llvm-project rev-parse --short=8 HEAD)" && \
    ARCH=$(uname -i) && \
    case "$ARCH" in \
      x86_64) ARCH_NAME="x64" ;; \
      aarch64) ARCH_NAME="arm64" ;; \
      *) ARCH_NAME="$ARCH" ;; \
    esac && \
    INSTALL_DIR="llvm-${SHORT_LLVM_COMMIT_HASH}-almalinux-${ARCH_NAME}" && \
    cmake -GNinja -Bllvm-project/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_ASM_COMPILER=clang \
    -DCMAKE_CXX_FLAGS="-Wno-everything" \
    -DCMAKE_LINKER=lld \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_EXECUTABLE=/usr/bin/python3.8 \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;lld" \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    llvm-project/llvm && \
    ninja -C llvm-project/build install && \
    tar czf ${INSTALL_DIR}.tar.gz ${INSTALL_DIR}

# Phase 3.1, collect ubuntu package
FROM scratch AS llvm_ubuntu

COPY --from=ubuntu_builder llvm*.tar.gz .


# Phase 3.2, collect almalinux package
FROM scratch AS llvm_almalinux

COPY --from=almalinux_builder llvm*.tar.gz .


# Phase 3.3 (default), collect all packages
FROM scratch AS llvm

COPY --from=ubuntu_builder llvm*.tar.gz .
COPY --from=almalinux_builder llvm*.tar.gz .

