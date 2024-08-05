#!/bin/bash
# export CUDA_VISIBLE_DEVICES="1"

 export HF_DATASETS_OFFLINE=1;
export HF_ENDPOINT=https://hf-mirror.com;
export TRANSFORMERS_OFFLINE=1;
# 定义模型名称数组
model_names=(

    # layer from 1-24
    "mobilevlm_v2_1.7b_20240722_120221/mobilevlm-linear-1.finetune"
    "mobilevlm_v2_1.7b_20240723_223338/mobilevlm-linear-2.finetune"
    "mobilevlm_v2_1.7b_20240721_185730/mobilevlm-linear-3.finetune"
    "mobilevlm_v2_1.7b_20240725_133137/mobilevlm-linear-4.finetune"
    "mobilevlm_v2_1.7b_20240725_220334/mobilevlm-linear-5.finetune"
    "mobilevlm_v2_1.7b_20240721_172927/mobilevlm-linear-6.finetune"
    "mobilevlm_v2_1.7b_20240726_111604/mobilevlm-linear-7.finetune"
    "mobilevlm_v2_1.7b_20240726_193949/mobilevlm-linear-8.finetune"
    "mobilevlm_v2_1.7b_20240721_102556/mobilevlm-linear-9.finetune"
    "mobilevlm_v2_1.7b_20240727_125901/mobilevlm-linear-10.finetune"
    "mobilevlm_v2_1.7b_20240727_220950/mobilevlm-linear-11.finetune"
    "mobilevlm_v2_1.7b_20240721_081558/mobilevlm-linear-12.finetune"
    "mobilevlm_v2_1.7b_20240728_064017/mobilevlm-linear-13.finetune"
    "mobilevlm_v2_1.7b_20240728_150420/mobilevlm-linear-14.finetune"
    "mobilevlm_v2_1.7b_20240722_044436/mobilevlm-linear-15.finetune"
    "mobilevlm_v2_1.7b_20240729_192100/mobilevlm-linear-16.finetune"
    "mobilevlm_v2_1.7b_20240724_161749/mobilevlm-linear-17.finetune"
    "mobilevlm_v2_1.7b_20240721_010307/mobilevlm-linear-18.finetune"
    "mobilevlm_v2_1.7b_20240723_130201/mobilevlm-linear-19.finetune"
    "mobilevlm_v2_1.7b_20240725_005434/mobilevlm-linear-20.finetune"
    "mobilevlm_v2_1.7b_20240722_145944/mobilevlm-linear-21.finetune"
    "mobilevlm_v2_1.7b_20240724_070214/mobilevlm-linear-22.finetune"
    "mobilevlm_v2_1.7b_20240722_233836/mobilevlm-linear-23.finetune"
    "mobilevlm_v2_1.7b_20240722_015504/mobilevlm-linear-24.finetune"
)


# 循环遍历模型名称并运行命令
for model_name in "${model_names[@]}"
do
    start_time=$(date +%s)  # 记录开始时间
    echo "Start eval: ${model_name}"

    python3 -m accelerate.commands.launch \
        --num_processes=4 \
        -m lmms_eval \
        --model mobilevlm \
        --model_args pretrained="/code/chr/MobileVLM/checkpoint/${model_name}" \
        --tasks refcoco \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "${model_name}" \
        --output_path "/code/chr/MobileVLM/logs/refcoco_1to24"

    end_time=$(date +%s)  # 记录结束时间
    elapsed_time=$((end_time - start_time))  # 计算运行时间
    echo "End eval: ${model_name}"
    echo "Model: $model_name took $elapsed_time seconds to run."
done