#!/bin/bash

# example usage: runbeaker.sh 1e-5 inputs outputs 2 5 5000

LR="$1"
DATADIR="$2"
NUM_EPOCH="$4"
BATCH_SIZE="$5"
NUM_TRAIN="$6"
WITH=${7:-' '}
FLAG=${8:-' '}
OUTPUT_PATH="$3/roberta_${LR}_${FLAG}${NUM_TRAIN}T${NUM_EPOCH}E"
METRICS=${9:-mu}
NUM_CHECKPOINTS=${10:-8}
if [[ "$WITH" == *"squad"* ]]; then
   SAVE_STEPS=$((2*$NUM_EPOCH*$NUM_TRAIN/$BATCH_SIZE/$NUM_CHECKPOINTS))
else
   SAVE_STEPS=$(($NUM_EPOCH*$NUM_TRAIN/$BATCH_SIZE/$NUM_CHECKPOINTS))
fi


echo "save steps: $SAVE_STEPS"
python3 roberta_fine-tuning.py  --data_dir $DATADIR/religion/randomsplit  --model_type robertaall  --model_name_or_path tli8hf/unqover-roberta-large-squad   --output_dir $OUTPUT_PATH --category religion  --learning_rate ${LR}  --overwrite_output_dir  --num_train_epochs $NUM_EPOCH  --per_gpu_train_batch_size $BATCH_SIZE  --num_train $NUM_TRAIN  --train $WITH --save_steps $SAVE_STEPS 

types=('religion')
for att in "${types[@]}"
do
   echo ${att}
   dir=$OUTPUT_PATH/$att
   if [ -d "$dir" ]; then
      echo "write to $dir"
   else
      echo "mkdir $dir"
      mkdir $dir
   fi
   python3 roberta_fine-tuning.py  --data_dir $DATADIR/${att}/randomsplit  --model_type  robertaall  --model_name_or_path $OUTPUT_PATH   --output_dir $dir  --category ${att}  --learning_rate ${LR}  --overwrite_output_dir  --num_train_epochs $NUM_EPOCH --per_gpu_train_batch_size $BATCH_SIZE  --num_train $NUM_TRAIN $WITH --save_steps $SAVE_STEPS --eval --eval_all_checkpoints

   for f in $dir/*.output.json
   do
      echo ${f}
      python3 ianalysis.py --input ${f}  --group_by subj | tee ${f:0:-12}.log.txt
   done
   python3 plot.py $dir $att $METRICS
done