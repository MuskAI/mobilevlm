LANGUAGE_MODEL=/code/chr/download/MobileLLaMA-1.4B-Base
VISION_MODEL=/model/BJiao/openaiclip-vit/openaiclip-vit-large-patch14-336
bash run_chr_24.sh mobilevlm_v2_1.7b pretrain-finetune ${LANGUAGE_MODEL} ${VISION_MODEL}
