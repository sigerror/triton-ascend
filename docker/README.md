# Build Guide: LLVM and Triton-Ascend for Ascend NPU

This guide provides instructions on how to build the Triton-Ascend project and its LLVM dependency, and how to package the build environment into Docker images for cross-platform development targeting Ascend NPUs.

---


## Clone Triton-Ascend

Make sure your Git version supports `--recurse-submodules`:

```shell
git clone https://gitee.com/ascend/triton-ascend.git --recurse-submodules --shallow-submodules
```



## Set Up Multi-Platform Build Environment (Optional)

Use Docker BuildKit (buildx) to enable multi-platform builds (e.g., amd64 and arm64):

```shell
docker buildx create --name mybuilder --use --driver docker-container

docker buildx inspect --bootstrap
```



## Build LLVM (Optional)

This step builds the LLVM toolchain required for the Triton compiler：

```shell
docker buildx build  --output type=local,dest=./ --platform linux/arm64 -f docker/build_llvm.dockerfile .
```

**Options**

* `--target llvm_ubuntu` : build LLVM based on Ubuntu 22.04
* `--target llvm_almalinux` : build LLVM based on AlmaLinux 8
* `--target llvm` : build LLVM based on both Ubuntu and AlmaLinux

> The output will be saved to the current directory.



## Build Triton

This builds the Triton compiler with support for multiple Python versions:

```shell
docker buildx build  --output type=local,dest=./ --platform linux/arm64 -f docker/build_triton.dockerfile .
```

**Options**

* `--build-arg BASE_IMAGE=quay.io/pypa/manylinux_2_28_aarch64` : specify the base image (manylinux_2_28_aarch64 or manylinux_2_28_x86_64)
* `--build-argPYTHON_ABIS="cp39 cp310 cp311` : specify Python versions to build for



## Build and Run Development Docker Image

Supports both arm64 and amd64 architectures. Useful for CI/CD or local development:

**Pull develop image from hub**

```shell
docker pull quay.io/ascend/triton:dev-cann8.2
```

**Build image manually (Optional)**

```shell
docker buildx build  --platform linux/arm64 -f docker/develop_env.dockerfile --load -t triton_ascend_dev .
```

**Run the Development Container**

```shell
docker run \
    --name triton_ascend \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it {image} bash
```

**Switch Python Version (Optional)**

```shell
sudo update-alternatives --set python /usr/bin/python3.10
```



## Notes

* Make sure your host system has the required Ascend driver stack and access permissions (device nodes, shared libraries, etc.).
* By default, this build process targets the linux/arm64 platform. Adjust the --platform flag for x86_64 as needed.
* If you encounter permission issues during buildx builds, prefer using the docker-container driver instead of the default docker driver.

