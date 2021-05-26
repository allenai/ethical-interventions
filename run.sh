#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python roberta_fine-tuning.py --train --data_dir ./../data/noIntvOverlap --model_type robertalarge --model_name_or_path roberta-large --output_dir ./../data/noIntvOverlap/robertalargeFT/ --eval --overwrite_output_dir --learning_rate 1e-5  --skiptrain 

# CUDA_VISIBLE_DEVICES=0 python roberta_fine-tuning.py --train --data_dir ./../data --model_type robertalarge --model_name_or_path tli8hf/unqover-roberta-large-squad  --output_dir robertalargesquadFT/  --eval --overwrite_output_dir  

