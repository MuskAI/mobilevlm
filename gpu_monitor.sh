#!/bin/bash
# Function to get GPU memory usage and utilization
LANGUAGE_MODEL=/code/chr/download/MobileLLaMA-1.4B-Base
VISION_MODEL=/model/BJiao/openaiclip-vit/openaiclip-vit-large-patch14-336

get_gpu_status() {
  # Extract memory usage and utilization from nvidia-smi output for the first GPU
  memory_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
  utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
  
  echo $memory_free $utilization
}

# Main loop to monitor GPU status
while true; do
  read memory_free utilization <<< $(get_gpu_status)
  echo $memory_free $utilization 
  # Check if memory is less than 100MB or utilization is less than 1%
  if (( memory_free > 70000 )) && (( utilization < 10 )); then
    bash run_chr_23.sh mobilevlm_v2_1.7b pretrain-finetune ${LANGUAGE_MODEL} ${VISION_MODEL}
    echo "Finished!"
    break
    
  fi
  echo "Monitoring GPU usage..."
  # Wait for 5 minutes before the next check
  sleep 600
done