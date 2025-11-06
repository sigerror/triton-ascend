# 总览说明

使用方式：`bash runCases.sh <TARGET_OPS_GROUP_NAME> <TEST_GROUP> <ARCHIVE_DIR>`

- `runCases.sh`，入口脚本，负责调起测试、日志分析
- 调用链：`pytest -> postProcessTestLog.sh -> countNumPassedFailedCases.py -> statErrors.py`
  - `postProcessTestLog.sh`，负责提取pytest日志，并解析为按照错误类型分类的总结
  - `countNumPassedFailedCases.py`，统计用例通过率
  - `statErrors.py`，汇总错误类型等信息

## `runCases.sh`

入口脚本，负责调起测试、日志分析。
用法：`bash runCases.sh <TARGET_OPS_GROUP_NAME> <TEST_GROUP> <ARCHIVE_DIR>`

命令行参数

- `TARGET_OPS_GROUP_NAME=[730|830]`，指定要测试的用例集，`runCases.sh`同级目录下有`test_ops.json`文件，该文件中是要测试的Op列表。730、830节点有对应的Op列表。
- `TEST_GROUP=[simple|full]`，要测试的测试集是否是全集，可选项有`[simple, full]`，`simple`表示精简集。比如`add`Op测试了完整shape、dtype，那么`sub, mul`等就只测试1个shape、dtype。
- `ARCHIVE_DIR=/home/shared/triton-regbase-daily-tests`，保存测试结果及日志的目录，一般不变。

脚本内设定参数解释如下：
- `NUM_THREADS=7`，pytest使用的线程数，理论上应该可以达到torchnpu设定的线程池中最大15个线程数，但是当前RegBased架构的bishengir-compile多线程编译有问题，解决后可扩展。
- `DEBUG=1`，是否保留cache等编译中间产物，`1`表示保留

其实运行只需要运行`runCases.sh`即可，需要再次开发改本套脚本才需要详细了解以下脚本。

## `postProcessTestLog.sh`

负责提取pytest日志，并解析为按照错误类型分类的总结。

用法：`bash postProcessTestLog.sh <log_filename> <op_name> <debug>`。
命令行参数：
- `log_filename`，需要处理的日志文件
- `op_name`，要测试的Op名，如`add`。会从`op_name`创建pytest测试脚本名`test_<op_name>_op.py`，从而开始pytest测试。
- `debug`，是否保留cache等编译中间产物，`1`表示保留

这个脚本内部执行了如下操作
- 精简pytest的日志，保留每个测试用例的必要测试信息
- 调用`organizeCasesByError.sh`，将精简的日志从按照测试用例及其错误信息的组织方式转换为错误类型及其对应的测试用例
- 调用`validateNumTestCases.sh`，验证原始日志信息在上述解析流程中没有遗漏或错误处理

## `countNumPassedFailedCases.py`

分析某个目录下所有`*.log`文件，统计用例通过率

用法：`python3 countNumPassedFailedCases.py <directory containing log files>`

命令行参数：
- `<directory conatining log files>`，包含上述日志`*.log`的目录

## `statErrors.py`

汇总错误类型等信息

用法：`python3 statErrors.py <directory containing summary log>`

命令行参数：
- `<directory containing summary log>`，包含上述`*_summary.log`日志，这个`*_summary.log`是`postProcessTestLog.sh`产生的，是对应`*.log`文件内容的总结。

# 定时执行

执行`crontab -e`后，使用如下命令

```
0 1 * * * bash /home/shared/contabs/cronRunTritonA5Cases.sh feature-bishengir-regbase >>/home/shared/crontabs/cronRunTritonA5Cases.log 2>&1
```

注意crontab中使用绝对路径。

`/home/shared/contabs/cronRunTritonA5Cases.sh`的样例内容如下

```
#!/bin/bash
# setup the env
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRART_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
. /opt/miniconda3/etc/profile.d/conda.sh
conda activate triton
# Update BiShengIR
cd <PATH to BiShengIR>/build
git pull origin feature-bishengir-regbase
ninja install
COMPILER_ROOT=<PATH to dir containing bishengir-compile>
export PATH=$COMPILER_ROOT:$COMPILER_ROOT/ccec_compiler/bin:$PATH
# Run Triton tests
TARGET_OPS_GROUP_NAME=730
TEST_GROUP=simple
ARCHIVE_DIR=/home/shared/triton-regbase-daily-tests
mkdir -p ${ARCHIVE_DIR}/$(date +%Y%m%d)
bash <PATH to runCases.sh> ${TARGET_OPS_GROUP_NAME} ${TEST_GROUP} ${ARCHIVE_DIR} 2>&1 | tee ${ARCHIVE_DIR}/$(date +%Y%m%d)/${TARGET_OPS_GROUP_NAME}.log
# Send message to Welink group: BiShengMAX Regbase
recv_gid=791007509686292884
auth=y30015927_RZp3HeFUIvb7hyGTks0SC1WMPlYfxcmX
python3 <PATH to sendMsgToWelink.py> ${ARCHIVE_DIR}/$(date +%Y%m%d)/${TARGET_OPS_GROUP_NAME}.log ${recv_gid} ${auth}
```

如果执行`crontab -e`后没有生效，注意查看`systemctl status cron`看是否cron服务没运行。
