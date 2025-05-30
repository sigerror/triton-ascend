# How to upload wheel to pypi

## Building Images

```shell
docker build --build-arg ARCH=aarch64 --build-arg LLVM_PATH=llvm-project --build-arg LLVM_COMMITID=b5cc222d7429fe6f18c787f633d5262fac2e676f --build-arg PYTHON_VERSION=310 -t manylinux-python310:latest .
```

## Buiding wheel

```shell
docker run -ti -d --privileged --rm manylinux-python310:latest bash
cd REPO_ROOT
./scripts/build.sh /root/triton-ascend/ascend/ /usr/local/llvm 0.0.1rc1 bdist_wheel
```

## Upload wheel to pypi test

```shell
twine upload --repository testpypi dist/triton_ascend-0.0.1rc1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```

## Upload wheel to pypi

```shell
twine upload dist/triton_ascend-0.0.1rc1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```
