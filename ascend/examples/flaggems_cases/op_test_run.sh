#!/bin/bash

# 脚本说明：该脚本根据参数控制，可执行正确性测试和性能测试，将脚本放至与FlagGems相同目录下执行
    # 正确性测试命令为：bash op_test_run.sh tests "abs,add"/"fullop"/"extop"
    # 性能测试命令为：bash op_test_run.sh benchmark "abs,add"/"fullop"/"extop"

# 判断是否传入了参数(算子测试区分功能正确性测试和性能测试两部分，需要传入参数区分测试哪部分)

# 定义正确性测试目录路径
DIR_TESTS="tests"
# 定义性能测试目录路径
DIR_BENCHMARK="benchmark"
# 正确性日志文件
LOG_NAME="result.json"

# 获取传入的参数
param="$1"
input_ops="$2"

OPS=("abs" "add" "addmm" "all" "amax" "arange" "argmax" "scaled_dot_product_attention" "apply_rotary_pos_emb" "bitwise_and" "bitwise_not" "bitwise_or" "bmm" "cat" "clamp" \
     "contiguous" "cos" "CrossEntropyLoss" "cumsum" "count_nonzero" "diag_embed" "diagonal_backward" "div" "dot" "dropout" "elu" "embedding" "eq" "erf" \
     "exp" "exponential_" "fill" "flip" "full_like" "full" "gather" "ge" "gelu" \
     "gelu_and_mul" "group_norm" "gt" "hstack" "index_select" "isinf" "isnan" "rsub" "le" "linspace" "log_sigmoid" "log" "logical_and" "logical_not" \
     "logical_or" "logical_xor" "lt" "masked_fill" "max" "maximum" "mean" "min" "minimum" "mm" "mse_loss" "mul" "mv" "native_dropout" "nan_to_num" "ne" \
     "neg" "normal" "ones_like" "pow" "prod" "rand_like" "rand" "randn_like" "randn" "reciprocal" "relu" "resolve_conj" "resolve_neg" "rms_norm" "rsqrt" \
     "select_scatter" "sigmoid" "silu" "silu_and_mul" "sin" "slice_scatter" "softmax" "stack" "sub" "sum" "tanh" "topk" "triu" "uniform_" "where" "threshold" \
     "zeros_like" "zeros")

OPS_EXT=( "instance_norm" "skip_rms_norm" "weight_norm" "angle" "any" "argmin" \
          "batch_norm" "conv_depthwise2d" "conv1d" "conv2d" "copy" "cummin" "diag" \
          "index_add" "index_put" "isclose" "isfinite" "isin" "outer" "kron" "layer_norm" \
          "log" "logical_and" "log_softmax" "masked_select" \
          "multinomial" "nllloss" "nonzero" "pad" "polar" "quantile" \
          "randperm" "repeat_interleave" "repeat" "sort" "tile" "unique" \
           "upsample_bicubic2d_aa" "upsample_nearest2d" "vdot" "vector_norm" "vstack" "var_mean")
if [ "$input_ops" == "fullop" ]; then
    OPS+=("${OPS_EXT[@]}")
elif [ "$input_ops" == "extop" ]; then
    OPS=("${OPS_EXT[@]}")
elif [[ -n "$input_ops" ]]; then
    OPS=()
    IFS=',' read -ra OPS <<< "$input_ops"  # 按逗号分割为数组
fi
echo "OPS: ${OPS[@]}"

# 根据参数进行不同测试执行
if [ "$param" == "tests" ]; then
    # 进入功能正确性测试目录
    cd "$DIR_TESTS" || { echo "无法进入目录 $DIR_TESTS"; exit 1; }

    # 循环执行Python脚本，每次替换一个算子
    for op_name in "${OPS[@]}"; do
        echo "正在执行: pytest -m $op_name --ref cpu -sv"
        # 全量用例
        pytest -m $op_name --ref cpu -sv
        # 单用例，将case_name修改成用例名称，如：test_unary_pointwise_ops.py::test_accuracy_isnan[dtype2-shape4]
        # pytest test_binary_pointwise_ops.py::test_accuracy_pow[dtype0-shape4] --ref cpu -sv
        # pytest -sv

        # 检查执行是否成功
        if [ $? -ne 0 ]; then
            echo "算子正确性执行失败，算子: $op_name"
        else
            echo "算子正确性执行成功，算子: $op_name"
        fi

        # 修改日志文件名称（由于每次执行会被覆盖，修改一下名称保留日志）
        if [ -f "$LOG_NAME" ]; then
            mv "$LOG_NAME" "result_${op_name}.json" && echo "文件重命名成功: $LOG_NAME -> result_${op_name}.json"
        else
            echo "文件不存在: $LOG_NAME"
            exit 1
        fi
    done
    echo "所有算子正确性测试执行完成！！！"

elif [ "$param" == "benchmark" ]; then
    # 进入目标目录
    cd "$DIR_BENCHMARK" || { echo "无法进入目录 $DIR_BENCHMARK"; exit 1; }

    # 循环执行Python脚本，每次替换一个参数
    for op_name in "${OPS[@]}"; do
        echo "正在执行: pytest -m $op_name --level core --record log"
        pytest -m $op_name --level core --record log

        # 检查执行是否成功
        if [ $? -ne 0 ]; then
            echo "性能执行失败，算子: $op_name"
        else
            echo "性能执行成功，算子: $op_name"
        fi
    done
    echo "所有算子性能测试执行完成！！！"
else
    echo "传入的参数错误，请重新传入正确参数进行测试！！！"
    exit 1
fi

