#!/bin/bash

# do model selection

DATADIR="$1"
RESDIR="$2"
NUM_TRAIN="$3"
BATCH_SIZE="$4"
NUM_GPUS="$5"
NUM_CHECKPOINTS=${6:-5}
WITH=${7:-'--doadversarial --doirrelevant --squad'}
FLAG=${8:-' '}
METRICS=${9:-mu}
for LR in 5e-6 1e-5 2e-5 3e-5
do 
   for NUM_EPOCH in 3 5 7 9
   do 
      OUTPUT_PATH="${RESDIR}/roberta_${FLAG}_${LR}_${NUM_TRAIN}T${NUM_EPOCH}E"
      SAVE_STEPS=$(($NUM_EPOCH*$NUM_TRAIN/$BATCH_SIZE/${NUM_GPUS}/$NUM_CHECKPOINTS)) #2gpus
      echo "save steps: $SAVE_STEPS"
      python3 roberta_fine-tuning.py  --data_dir $DATADIR/religion/noniids  --model_type roberta_${LR}_${NUM_EPOCH}  --model_name_or_path tli8hf/unqover-roberta-large-squad   --output_dir $OUTPUT_PATH --category religion  --learning_rate ${LR}  --overwrite_output_dir  --num_train_epochs $NUM_EPOCH  --per_gpu_train_batch_size $BATCH_SIZE  --num_train $NUM_TRAIN  --train $WITH --save_steps $SAVE_STEPS --eval --evaldev dev
   done
done

for f in ${RESDIR}/*/*-dev.output.json
do
   echo ${f}
   python3 ianalysis.py --input ${f}  --group_by subj | tee ${f:0:-12}.log.txt
done
best=($(python3 model_select.py ${RESDIR}))
BEST_LR=${best[0]}
BEST_EPOCH=${best[1]}
echo "BEST_LR:${BEST_LR}; BEST_EPOCH:${BEST_EPOCH}"

#do eval with the best model on the noniid_test.

types=('religion' 'ethnicity' 'gender')
for att in "${types[@]}"
do
   echo ${att}
   dir=${RESDIR}/roberta_${FLAG}_${BEST_LR}_${NUM_TRAIN}T${BEST_EPOCH}E/$att
   if [ -d "$dir" ]; then
      echo "write to $dir"
   else
      echo "mkdir $dir"
      mkdir $dir
   fi
   python3 roberta_fine-tuning.py  --data_dir $DATADIR/${att}/noniids  --model_type  roberta_${BEST_LR}_${BEST_EPOCH}  --model_name_or_path ${RESDIR}/roberta_${FLAG}_${BEST_LR}_${NUM_TRAIN}T${BEST_EPOCH}E --output_dir $dir  --category ${att}  --learning_rate ${BEST_LR}  --overwrite_output_dir  --num_train $NUM_TRAIN --per_gpu_train_batch_size $BATCH_SIZE  --eval  --evaldev test


   for f in $dir/*-test.output.json
   do
      echo ${f}
      python3 ianalysis.py --input ${f}  --group_by subj | tee ${f:0:-12}.log.txt
   done
   # python3 plot.py $dir $att $METRICS
done

types=('religion')
for att in "${types[@]}"
do
   echo "do all checkpoint eval for ${att}"
   dir=${RESDIR}/roberta_${FLAG}_${BEST_LR}_${NUM_TRAIN}T${BEST_EPOCH}E/$att
   if [ -d "$dir" ]; then
      echo "write to $dir"
   else
      echo "mkdir $dir"
      mkdir $dir
   fi
   python3 roberta_fine-tuning.py  --data_dir $DATADIR/${att}/noniids  --model_type  roberta_${LR}_${NUM_EPOCH}  --model_name_or_path ${RESDIR}/roberta_${FLAG}_${BEST_LR}_${NUM_TRAIN}T${BEST_EPOCH}E --output_dir $dir  --category ${att}  --learning_rate ${BEST_LR}  --overwrite_output_dir  --num_train $NUM_TRAIN --per_gpu_train_batch_size $BATCH_SIZE  --eval --eval_all_checkpoints --evaldev dev


   for f in $dir/*-dev*.output.json
   do
      echo ${f}
      python3 ianalysis.py --input ${f}  --group_by subj | tee ${f:0:-12}.log.txt
   done
   python3 plot.py $dir $att $METRICS
done