# 说明

- `buildDocker.sh`：构建docker image的脚本
  - 要求提前准备四个文件到当前目录下：
    - `bishengir-compile`：你自己构建好的RegBased架构的bishengir-compile
    - `bisheng`：你自己构建好或者从CMC下载的bisheng，可以被RegBased架构的bishengir-compile调用
    - `ld.lld`：你自己构建好或者从CMC下载的ld.lld，一般跟bisheng同目录
    - `id_rsa`：可以从codehub代码仓pull代码的ssh private key，可以从你自己的ssh目录中拿过来，dockerfile中使用后会删除。
- `triton-dev-outofbox.dockerfile`：镜像文件
- `saveDockerImage.sh`：用于将构建好的docker image保存为文件，后续其他人可以导入到其他服务器使用

# 使用方法

- 准备好
- 执行`buildDocker.sh <proxy_url>`进行构建，`<proxy_url>`按照`http://工号:转义后的密码@proxyhk.huawei.com:8080`格式输入。
