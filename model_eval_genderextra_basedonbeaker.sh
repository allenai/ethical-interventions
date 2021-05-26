#!/bin/bash

#re-evaluate the model based on the beaker predictions 

GENDEREXTRADATA="$1"
OUTPUTS="$2"
NUM_TRAIN="$3"
BATCH_SIZE="$4"
NUM_GPUS="$5"
NUM_CHECKPOINTS=${6:-5}
WITH=${7:-'--doadversarial --doirrelevant --squad'}
FLAG=${8:-' '}
METRICS=${9:-mu}
BEAKERSAVED=${10}
TOP5=${11}

genders=("Female" "Male")
for gender in ${genders[@]}
do
   for top in ${TPO5[@]}
   do 
      f=${BEAKERSAVED}/${top}
      echo ${f}
      name=${top}
      echo "evaluating ${gender} extra on model ${top}"
      python3 -u -m qa_hf.predict --gpuid 0 --hf_model ${f} --intv_type none --input ${GENDEREXTRADATA}/extra${gender}withIntvs-sample.source.json --output ${OUTPUTS}/extra${gender}withIntvs-sample-${name}.output.json --attribute gender --converted 0

      python3 analysis.py --input ${OUTPUTS}/extra${gender}withIntvs-sample-${name}.output.json --metrics subj_bias,model --group_by subj | tee ${OUTPUTS}/extra${gender}withIntvs-${name}.log.txt
   done
done