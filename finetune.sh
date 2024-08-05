LANGUAGE_MODEL=/code/chr/download/MobileLLaMA-1.4B-Base
VISION_MODEL=/model/BJiao/openaiclip-vit/openaiclip-vit-large-patch14-336
OUTPUT_DIR=/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240721_185730
bash run_chr_4.sh mobilevlm_v2_1.7b finetune ${LANGUAGE_MODEL} ${VISION_MODEL} ${OUTPUT_DIR}

# run_chr.sh mobilevlm_v2_1.7b pretrain /code/chr/download/MobileVLM_V2-1.7B /model/BJiao/openaiclip-vit/openaiclip-vit-large-patch14-336