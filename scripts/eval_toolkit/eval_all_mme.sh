#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
# 定义模型名称数组
model_names=(
    # "mobilevlm_v2_1.7b_20240728_231609/mobilevlm-mlp2x_gelu-23.finetune"
    # "mobilevlm_v2_1.7b_20240722_233836/mobilevlm-linear-23.finetune"
    # "mobilevlm_v2_1.7b_20240729_120439/mobilevlm-c-abstractor-23.finetune"
    # "mobilevlm_v2_1.7b_20240729_143803/mobilevlm-mlp2x_gelu-23.finetune"
    "mobilevlm_v2_1.7b_20240730_101520/mobilevlm-c-abstractor-23.finetune"
    
)


# 循环遍历模型名称并运行命令
for model_name in "${model_names[@]}"
do
    start_time=$(date +%s)  # 记录开始时间
    ./chr_eval_all.sh "$model_name" mme
    end_time=$(date +%s)  # 记录结束时间
    elapsed_time=$((end_time - start_time))  # 计算运行时间
    echo "Model: $model_name took $elapsed_time seconds to run."
done