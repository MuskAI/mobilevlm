#!/bin/bash

# Check if MODEL_NAME is provided as a command line argument
if [ -z "$1" ]; then
  echo "Usage: $0 <MODEL_NAME> <DATASET1> <DATASET2> ..."
  exit 1
fi

# Set the model name and checkpoint
MODEL_NAME="$1"
CKPT="$MODEL_NAME"

# Shift the arguments so that $@ only contains datasets
shift

# 遍历并打印每个参数
for param in "$@"; do
  echo "Ready to eval datasets: $param"
done

for param in "$@"; do
  echo "Now we are evaluating: $param"
  if [ "$param" = "textvqa" ]; then
    echo "The parameter is textvqa."

    ######################## 1 text vqa ###############################
    python -m mobilevlm.eval.model_vqa_loader \
        --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
        --question-file /data/JoeyLin/llava/llava1-5/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder /data/JoeyLin/llava/llava1-5/data/eval/textvqa/train_images \
        --answers-file /code/chr/MobileVLM/eval_results/textvqa/answers/${MODEL_NAME}.jsonl \
        --temperature 0 \
        --conv-mode v1 &&
    sleep 5;

    python -m llava.eval.eval_textvqa \
        --annotation-file /data/JoeyLin/llava/llava1-5/data/eval/textvqa/TextVQA_0.5.1_val.json \
        --result-file /code/chr/MobileVLM/eval_results/textvqa/answers/${MODEL_NAME}.jsonl
    #################################################################
  fi

  if [ "$param" = "pope" ]; then
    ########################## 2 pope ###############################
    python -m mobilevlm.eval.model_vqa_loader \
        --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
        --question-file /data/JoeyLin/llava/llava1-5/data/eval/pope/llava_pope_test.jsonl \
        --image-folder /data/JoeyLin/llava/llava1-5/data/eval/pope/val2014 \
        --answers-file /code/LLaVA-unet/LLaVA_orign/eval_results/pope/answers/checkpoints/${MODEL_NAME}.jsonl \
        --temperature 0 \
        --conv-mode v1

    python /code/LLaVA-unet/LLaVA_orign/llava/eval/eval_pope.py \
        --annotation-dir /data/JoeyLin/llava/llava1-5/data/eval/pope/coco \
        --question-file /data/JoeyLin/llava/llava1-5/data/eval/pope/llava_pope_test.jsonl \
        --result-file /data/JoeyLin/llava/llava1-5/data/eval/pope/answers/checkpoints/${MODEL_NAME}.jsonl
    #################################################################
  fi

  if [ "$param" = "vqav2" ]; then
    gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
    IFS=',' read -ra GPULIST <<< "$gpu_list"

    CHUNKS=1
    SPLIT="llava_vqav2_mscoco_test-dev2015"
    ########################## 3 vqa ###############################
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mobilevlm.eval.model_vqa_loader \
            --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
            --question-file /data/JoeyLin/llava/llava1-5/data/eval/vqav2/$SPLIT.jsonl \
            --image-folder /data/JoeyLin/llava/llava1-5/data/test2015 \
            --answers-file  /code/LLaVA-unet/LLaVA_orign/eval_results/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode v1 &
    done

    wait

    output_file=/code/LLaVA-unet/LLaVA_orign/eval_results/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat /code/LLaVA-unet/LLaVA_orign/eval_results/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
    
    python /code/LLaVA-unet/LLaVA_orign/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT
    #################################################################
  fi

  if [ "$param" = "scienceqa" -o "$param" = "sqa" ]; then
    ######################### 4 sqa ###############################
    python -m llava.eval.model_vqa_science \
        --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
        --question-file  /data/JoeyLin/llava/llava1-5/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder  /data/JoeyLin/llava/llava1-5/data/eval/scienceqa/test \
        --answers-file  /code/LLaVA-unet/LLaVA_orign/eval_results/scienceqa/answers/${MODEL_NAME}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode v1 &&

    python  /code/LLaVA-unet/LLaVA_orign/llava/eval/eval_science_qa.py \
        --base-dir  /data/JoeyLin/llava/llava1-5/data/eval/scienceqa \
        --result-file  /code/LLaVA-unet/LLaVA_orign/eval_results/scienceqa/answers/${MODEL_NAME}.jsonl \
        --output-file  /code/LLaVA-unet/LLaVA_orign/eval_results/scienceqa/answers/${MODEL_NAME}_output.jsonl \
        --output-result  /code/LLaVA-unet/LLaVA_orign/eval_results/scienceqa/answers/${MODEL_NAME}_result.json
    #################################################################
  fi

  if [ "$param" = "gqa" ]; then
    ########################## 5 gqa ###############################
    gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
    IFS=',' read -ra GPULIST <<< "$gpu_list"

    CHUNKS=${#GPULIST[@]}
    CKPT="${MODEL_NAME}"
    SPLIT="llava_gqa_testdev_balanced"
    GQADIR="/data/JoeyLin/llava/llava1-5/data/eval/gqa"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mobilevlm.eval.model_vqa_loader \
            --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
            --question-file /data/JoeyLin/llava/llava1-5/data/eval/gqa/$SPLIT.jsonl \
            --image-folder /data/JoeyLin/llava/llava1-5/data/eval/gqa/images \
            --answers-file /code/LLaVA-unet/LLaVA_orign/eval_results/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode v1
    done

    wait

    output_file=/code/LLaVA-unet/LLaVA_orign/eval_results/gqa/answers/$SPLIT/$CKPT/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat /code/LLaVA-unet/LLaVA_orign/eval_results/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python /code/LLaVA-unet/LLaVA_orign/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions_${MODEL_NAME}.json
    cd $GQADIR
    # python /data/JoeyLin/llava/llava1-5/data/eval/gqa/eval.py --tier /data/JoeyLin/llava/llava1-5/data/eval/gqa/testdev_balanced
    #################################################################
  fi

  if [ "$param" = "mme" ]; then
    ######################## 6 mme ###############################
    # python -m mobilevlm.eval.model_vqa_loader \
    #    --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
    #    --question-file  /data/JoeyLin/llava/llava1-5/data/eval/MME/llava_mme.jsonl \
    #    --image-folder  /data/JoeyLin/llava/llava1-5/data/eval/MME/MME_Benchmark_release_version \
    #    --answers-file  /code/chr/MobileVLM/eval_results/MME/answers/${MODEL_NAME}.jsonl \
    #    --temperature 0 \
    #    --conv-mode v1 &&

    cd  /code/chr/MobileVLM/eval_results/MME
    # python /code/chr/MobileVLM/eval_results/MME/convert_answer_to_mme.py --experiment ${MODEL_NAME};

    cd eval_tool
    python  /code/chr/MobileVLM/eval_results/MME/eval_tool/calculation.py --results_dir  /code/chr/MobileVLM/eval_results/MME/eval_tool/answers/${MODEL_NAME}

 
    #################################################################
  fi

  if [ "$param" = "mmbench" ]; then
    ########################## 7 mmbench ###############################
    SPLIT="mmbench_dev_20230712"

    python -m mobilevlm.eval.model_vqa_mmbench \
        --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
        --question-file /data/JoeyLin/llava/llava1-5/data/eval/mmbench/$SPLIT.tsv \
        --answers-file /code/chr/MobileVLM//eval_results/mmbench/answers/$SPLIT/${MODEL_NAME}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode v1 &&
  
    mkdir -p /code/chr/MobileVLM/eval_results/mmbench/answers_upload/${MODEL_NAME};

    python  /code/chr/MobileVLM/scripts/convert_mmbench_for_submission.py \
        --annotation-file /data/JoeyLin/llava/llava1-5/data/eval/mmbench/$SPLIT.tsv \
        --result-dir /code/chr/MobileVLM/eval_results/mmbench/answers/$SPLIT/${MODEL_NAME} \
        --upload-dir /code/chr/MobileVLM/eval_results/mmbench/answers_upload/${MODEL_NAME} \
        --experiment ${MODEL_NAME}
    #################################################################
  fi

  if [ "$param" = "cvbench" ]; then
    ######################## 6 mme ###############################
    python -m mobilevlm.eval.model_vqa_cvbench \
       --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
       --question-file  /code/chr/download/CV-Bench/test.jsonl \
       --image-folder  /code/chr/download/CV-Bench \
       --answers-file  /code/chr/MobileVLM/eval_results/CVBench/answers/${MODEL_NAME}.jsonl \
       --temperature 0 \
       --conv-mode v1 &&

    python /code/chr/MobileVLM/chr_toolkits/cvbench.py \
        --json_path /code/chr/MobileVLM/eval_results/CVBench/answers/${MODEL_NAME}.jsonl \
        --result_path /code/chr/MobileVLM/eval_results/CVBench/answers/${MODEL_NAME}.txt

  fi

  if [ "$param" = "mmvet" ]; then
    python -m mobilevlm.eval.model_vqa_mmvet \
      --model-path /code/chr/MobileVLM/checkpoint/${MODEL_NAME} \
      --question-file /data/JoeyLin/llava/llava1-5/data/eval/mm-vet/llava-mm-vet.jsonl \
      --image-folder /code/chr/download/mm-vet/images \
      --answers-file /code/chr/MobileVLM/eval_results/mm-vet/answers/${MODEL_NAME} \
      --temperature 0 \
      --conv-mode v1 &&

    mkdir -p /code/chr/MobileVLM/eval_results/mm-vet/results

    python /code/chr/MobileVLM/scripts/convert_mmvet_for_eval.py \
        --src /code/chr/MobileVLM/eval_results/mm-vet/answers/${MODEL_NAME}.jsonl \
        --dst /code/chr/MobileVLM/eval_results/mm-vet/results/${MODEL_NAME}.json

  fi


done