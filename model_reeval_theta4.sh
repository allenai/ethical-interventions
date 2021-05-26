#!/bin/bash

# do model selection

DATADIR="$1"
OUTPUTS="$2"
NUM_TRAIN="$3"
BATCH_SIZE="$4"
NUM_GPUS="$5"
NUM_CHECKPOINTS=${6:-5}
WITH=${7:-'--doadversarial --doirrelevant --squad'}
FLAG=${8:-' '}
METRICS=${9:-mu}
WEIGHT_ETHICAL=${10:-1}
WEIGHT_ADVERSARIAL=${11:-1}
WEIGHT_IRRELEVANT=${12:-1}
for LR in 5e-6 1e-5 2e-5 3e-5
do 
   for NUM_EPOCH in 3 5 7 9
   do 
      OUTPUT_PATH="${OUTPUTS}/roberta_${FLAG}_${LR}_${NUM_TRAIN}T${NUM_EPOCH}E"
      SAVE_STEPS=$(($NUM_EPOCH*$NUM_TRAIN/$BATCH_SIZE/${NUM_GPUS}/$NUM_CHECKPOINTS)) #2gpus
      echo "save steps: $SAVE_STEPS"
      python3 roberta_fine-tuning.py  --data_dir $DATADIR/religion/noniids  --model_type roberta_${LR}_${NUM_EPOCH}  --model_name_or_path tli8hf/unqover-roberta-large-squad   --output_dir $OUTPUT_PATH --category religion  --learning_rate ${LR}  --overwrite_output_dir  --num_train_epochs $NUM_EPOCH  --per_gpu_train_batch_size $BATCH_SIZE  --num_train $NUM_TRAIN --train $WITH --save_steps $SAVE_STEPS --eval --evaldev dev --weight_ethical ${WEIGHT_ETHICAL} --weight_adversarial ${WEIGHT_ADVERSARIAL} --weight_irrelevant ${WEIGHT_IRRELEVANT}
   done
done

for f in ${OUTPUTS}/*/*-dev.output.json
do
   echo ${f}
   python3 ianalysis.py --input ${f}  --group_by subj | tee ${f:0:-12}.log.txt
done
# python3 model_select.py ${OUTPUTS} ${NUM_TRAIN} 
best=($(python3 model_select.py ${OUTPUTS} ${NUM_TRAIN} ${METRICS}))
# BEST_LR=${best[0]}
# BEST_EPOCH=${best[1]}
# echo "BEST_LR:${BEST_LR}; BEST_EPOCH:${BEST_EPOCH}"
TOP1=${best[0]}
echo "top-5 BEST MODELS:${best[@]}" #a list of best models
echo "top-1 BEST MODEL:${TOP1}"

#do eval with the best model on the noniid_test.

types=('religion' 'ethnicity' 'gender')
for att in "${types[@]}"
do
   echo ${att}
   for topk in ${best[@]} #do eval for top5 models
   do
      dir=${OUTPUTS}/${topk}/$att
      if [ -d "$dir" ]; then
         echo "write to $dir"
      else
         echo "mkdir $dir"
         mkdir $dir
      fi
      python3 roberta_fine-tuning.py  --data_dir $DATADIR/${att}/noniids  --model_type  ${topk}  --model_name_or_path ${OUTPUTS}/${topk} --output_dir $dir  --category ${att}  --overwrite_output_dir  --num_train $NUM_TRAIN --per_gpu_train_batch_size $BATCH_SIZE  --eval  --evaldev test
      if [[ "$att" == "religion" ]]; then
          echo "do squad eval"
          python3 transformers/examples/question-answering/run_squad.py   --model_type ${topk}      --do_eval      --train_file $DATADIR/squad/train-v1.1.json   --predict_file $DATADIR/squad/dev-v1.1.json   --per_gpu_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2.0   --max_seq_length 384   --doc_stride 128  --seed 3435 --do_lower_case --overwrite_cache --output_dir /tmp/debug_squad/${topk}   --model_name_or_path ${OUTPUTS}/${topk}
      fi
   done

   for f in ${OUTPUTS}/*/${att}/*-test.output.json
   do
      echo ${f}
      python3 ianalysis.py --input ${f}  --group_by subj | tee ${f:0:-12}.log.txt
   done
   # python3 plot.py $dir $att $METRICS
done

#below is to draw the \mu for all checkpoints; ignored for now.
# types=('religion')
# for att in "${types[@]}"
# do
#    echo "do all checkpoint eval for ${att}"
#    dir=${OUTPUTS}/roberta_${FLAG}_${BEST_LR}_${NUM_TRAIN}T${BEST_EPOCH}E/$att
#    if [ -d "$dir" ]; then
#       echo "write to $dir"
#    else
#       echo "mkdir $dir"
#       mkdir $dir
#    fi
#    python3 roberta_fine-tuning.py  --data_dir $DATADIR/${att}/noniids  --model_type  roberta_${LR}_${NUM_EPOCH}  --model_name_or_path ${RESDIR}/roberta_${FLAG}_${BEST_LR}_${NUM_TRAIN}T${BEST_EPOCH}E --output_dir $dir  --category ${att}  --learning_rate ${BEST_LR}  --overwrite_output_dir  --num_train $NUM_TRAIN --per_gpu_train_batch_size $BATCH_SIZE  --eval --eval_all_checkpoints --evaldev dev


#    for f in $dir/*-dev*.output.json
#    do
#       echo ${f}
#       python3 ianalysis.py --input ${f}  --group_by subj | tee ${f:0:-12}.log.txt
#    done
#    python3 plot.py $dir $att $METRICS
# done