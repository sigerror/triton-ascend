# Triton-Ascend 开发者镜像使用文档

- ## 镜像说明
用户在 [下载链接](https://quay.io/repository/ascend/triton?tab=tags) 中可以获取最新的下载地址，该链接中的镜像是代码仓执行流水线的镜像，会同步更新，用户取最新的镜像即可。

- ## 前提工作
用户在宿主机上需提前安装好昇腾卡的驱动与firmware。

- ## 拉取镜像
用户可以在 [镜像链接](https://quay.io/repository/ascend/triton?tab=tags) ，通过Fetch Tag获取镜像拉取命令，例如：
```shell
docker pull quay.io/ascend/triton:dev-cd3d223
```
- ## 生成容器
启动容器命令参考如下：
```shell
docker run \
  --name xxx_container \
  -d \
  -v /path/to/map:/path/to/map \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -e ASCEND_RUNTIME_OPTIONS=NODRV --privileged=true \
  -it quay.io/ascend/triton:dev-cd3d223 /bin/bash
```
**注意**：xxx_container，/path/to/map和dev-cd3d223请根据实际修改，并且注意不要映射/home目录。实际映射的宿主机的路径请用户自行检查是否存在。
- ## 启动容器
启动容器命令参考如下：
```shell
docker exec -it xxx_container /bin/bash
```
- ## 编译源码
克隆代码仓：
```shell
git clone https://gitcode.com/Ascend/triton-ascend.git --recurse-submodules --shallow-submodules
```
编译源码命令参考如下：
```shell
cd triton-ascend && make package PYTHON=$PY IS_MANYLINUX=False 
```
PY的取值为python3.9,python3.10,python3.11其中之一。
