MODEL_LOADER=mobilevlm.eval.model_vqa_loader
MODEL_DIR=/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240720_234344/mobilevlm-linear-23.finetune
CONV_MODE=v1
SPLIT_MME=llava_mme
DATA_ROOT=/data/JoeyLin/llava/llava1-5/data/eval/MME
SAVE_PATH=/code/chr/MobileVLM/eval_results/${SPLIT_MME}

bash /code/chr/MobileVLM/scripts/benchmark/mme.sh ${MODEL_LOADER} ${MODEL_DIR} ${CONV_MODE} ${SPLIT_MME} ${DATA_ROOT} ${SAVE_PATH}