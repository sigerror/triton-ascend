ARG BASE_IMAGE=quay.io/pypa/manylinux_2_28_aarch64
FROM ${BASE_IMAGE} AS builder
ARG PYTHON_ABIS="cp39 cp310 cp311"

RUN dnf install --assumeyes clang cmake lld ninja-build

WORKDIR /triton-ascend
COPY . .

# Prepare llvm binray if exists.
RUN if ls llvm*.tar.gz 1>/dev/null 2>&1; then \
      TAR_FILE=$(ls llvm*.tar.gz); \
      tar -xzf "$TAR_FILE"; \
      EXTRACTED_DIR=$(tar -tzf "$TAR_FILE" | head -1 | cut -f1 -d"/"); \
      rm -f "$TAR_FILE"; \
      mv "$EXTRACTED_DIR" llvm_install; \
    fi

# Build
RUN for python_ver in ${PYTHON_ABIS}; do \
        PATH=/opt/python/${python_ver}-${python_ver}/bin:$PATH; \
        echo "Python Version = $(python3 --version)" && \
        pip3 install \
        auditwheel \
        patchelf \
        twine && \
        pip3 install -r requirements_dev.txt && \
        pip3 install -r requirements.txt && \
        if [ -d llvm_install ]; then \
          LLVM_SYSPATH=llvm_install; \
        fi && \
        TRITON_PLUGIN_DIRS=./ascend \
        TRITON_BUILD_WITH_CLANG_LLD=true \
        TRITON_BUILD_PROTON=OFF \
        TRITON_WHEEL_NAME="triton-ascend" \
        TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
        python3 setup.py bdist_wheel && \
        python3 setup.py clean; \
    done

# Collect triton_ascend wheels
FROM scratch AS triton

COPY --from=builder /triton-ascend/dist/*.whl .
