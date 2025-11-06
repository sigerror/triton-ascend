# docker build -t triton-dev-llvm:latest -f triton-dev-llvm.dockerfile .

FROM triton-dev-base:latest

ARG TARGET_COMMIT=b5cc222d7429fe6f18c787f633d5262fac2e676f

COPY . .

# TODO: upload to gitee or huawei-yellow intranet. Then we only need to download it.
RUN if [ ! -d llvm-project ]; then \
        if [ -e llvm-project.tar.gz ]; then \
            echo "extracing LLVM source codes..." && \
            tar xf llvm-project.tar.gz; \
            rm llvm-project.tar.gz; \
        else \
            echo "cloning LLVM repo..." && \
            git clone --depth 1 https://github.com/llvm/llvm-project.git; \
        fi \
    fi

RUN cd llvm-project && \
    echo "checking LLVM commit..." && \
    CURRENT_COMMIT=$(git rev-parse HEAD) && \
    if [ "$CURRENT_COMMIT" != "$TARGET_COMMIT" ]; then \
        echo "LLVM is not $TARGET_COMMIT, changing..." && \
        git fetch --depth 1 origin $TARGET_COMMIT && \
        git checkout $TARGET_COMMIT; \
    else \
        echo "LLVM is already $TARGET_COMMIT"; \
    fi

RUN if [ ! -d /opt/llvm-$TARGET_COMMIT/lib ]; then \
        echo "Start building LLVM $TARGET_COMMIT..." && \
        mkdir -p llvm-project/build && \
        cd llvm-project/build && \
        /opt/cmake/bin/cmake ../llvm \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_ENABLE_ASSERTIONS=ON \
            -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
            -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
            -DCMAKE_INSTALL_PREFIX=/opt/llvm-$TARGET_COMMIT \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DLLVM_ENABLE_LLD=ON && \
        ninja install && \
        # 清理构建文件以减小镜像体积
        cd - && \
        rm -rf llvm-project; \
    else \
        echo "LLVM $TARGET_COMMIT is installed"; \
    fi
