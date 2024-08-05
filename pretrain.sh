LANGUAGE_MODEL=/code/chr/download/MobileVLM_V2-1.7B
VISION_MODEL=/model/BJiao/openaiclip-vit/openaiclip-vit-large-patch14-336
OUTPUT_DIR=/code/chr/MobileVLM/checkpoint
bash run_chr.sh mobilevlm_v2_1.7b pretrain ${LANGUAGE_MODEL} ${VISION_MODEL} ${OUTPUT_DIR}

# run_chr.sh mobilevlm_v2_1.7b pretrain /code/chr/download/MobileVLM_V2-1.7B /model/BJiao/openaiclip-vit/openaiclip-vit-large-patch14-336