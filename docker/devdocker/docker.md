#  triton_ascend运行态容器镜像文档说明书

该文档给出了triton_ascend以容器形式进行运行态环境的部署。宿主机要求：昇腾Ascend 910AI卡。支持aarch64和x86_64架构。

相关脚本及功能说明：

| 脚本 | 功能 |
| :----: | :----: |
| triton-ascend_dev.dockerfile | 构建镜像的dockerfile |
| set_triton-ascend_dev.sh | 设置容器运行时的环境 |
| build_docker.sh | 构建镜像 |
| run_docker.sh | 启动容器 |

构建步骤：

1. 将devdocker目录拷贝至宿主机，在devdocker目录下新建packages目录，并按照[README.md](../../README.md)中```安装Ascend CANN``` 章节下载对应版本的toolkit和kernel包至packages目录。
1. 运行命令```bash build_docker.sh``` ，该命令会根据宿主机的架构生成对应的image，生成的镜像名称默认为triton-ascend_dev_${ARCH}:1.0，如需更改版本号，请在build_docker.sh中自行修改。
2. 运行命令```bash run_docker.sh``` ，该命令会启动一个container。注意```-v /data/disk```为宿主机和container的共享目录，请根据宿主机目录及实际需要设置；```-name triton_dev``` 为启动的container名称，请按需修改；```-it triton-ascend_build_${ARCH}:1.0```  为第一步生成的镜像，请按实际修改名称及版本号。
3. 运行命令```docker ps``` 获取实际的container名称，该容器名称实际为第二步```-name ``` 指定的名称。运行``` docker exex -it triton_dev bash``` 进入容器，请将```trtion_dev``` 替换为实际的容器名称。
4.  进入容器后，切换目录到```/home/docker_triton-ascend_dev``` ,运行命令``` chmod 700 set_triton-ascend_dev.sh && bash set_triton-ascend_dev.sh ``` 。该命令会会在容器里启动```py311``` ，```py310``` , ```py39``` 三个conda环境，并装好triton_ascend需要的所有依赖，然后拉取最新的master分支进行一次编译安装。安装的默认版本为0.0.1，如有修改请在```set_triton-ascend_dev.sh``` 中修改。



