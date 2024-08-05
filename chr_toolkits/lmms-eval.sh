export CUDA_VISIBLE_DEVICES="3";
export HF_DATASETS_OFFLINE=1;
export HF_ENDPOINT=https://hf-mirror.com;
export TRANSFORMERS_OFFLINE=1;

python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model mobilevlm \
    --model_args pretrained="/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240722_233836/mobilevlm-linear-23.finetune" \
    --tasks seedbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mobilevlm_v2_1.7b_20240722_233836_mobilevlm-linear-23 \
    --output_path /code/chr/MobileVLM/logs/